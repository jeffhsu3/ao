# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.uintx.bitpacking import pack, unpack
from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl
from torchao.dtypes.utils import Layout


@dataclass(frozen=True)
class Int4PlainLayout(Layout):
    """Layout class for int4 plain layout for affine quantized tensor.
    Stores mathematically flat int4 data efficiently packed into a uint8 tensor.
    """

    pack_dim: int = -1
    pad_amount: int = 0

    def post_process(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return input, scale, zero_point


@register_layout(Int4PlainLayout)
class Int4PlainAQTTensorImpl(PlainAQTTensorImpl):
    """
    TensorImpl for Int4 plain layout. Extends PlainAQTTensorImpl.
    Uses bitpacking to store 2 int4 elements per uint8 byte in `int_data`.
    The `scale` and `zero_point` are stored unchanged as independent PyTorch tensors.
    """

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, Int4PlainLayout)
        int_data_uint8 = int_data.to(torch.uint8)

        # Pad if needed. bitpacking.pack requires the packed dimension to be a multiple of 8
        dim = _layout.pack_dim
        if dim < 0:
            dim += int_data.dim()

        pad_amount = (8 - (int_data.shape[dim] % 8)) % 8
        if pad_amount > 0:
            pad_tuple = [0, 0] * int_data.dim()
            pad_tuple[2 * (int_data.dim() - 1 - dim) + 1] = pad_amount
            int_data_uint8 = torch.nn.functional.pad(int_data_uint8, pad_tuple)

        # Recreate layout with padding amount
        _layout = Int4PlainLayout(pack_dim=_layout.pack_dim, pad_amount=pad_amount)

        # Pack the 4-bit elements (stored in uint8) securely, slicing dim in half
        shards = pack(int_data_uint8, 4, dim=_layout.pack_dim)
        packed_int_data = shards[0]
        return cls(packed_int_data, scale, zero_point, _layout)

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Dynamically unpack for compute backends or fallbacks
        unpacked_int_data = unpack([self.int_data], 4, dim=self._layout.pack_dim)

        # Unpad if needed
        if self._layout.pad_amount > 0:
            dim = self._layout.pack_dim
            if dim < 0:
                dim += unpacked_int_data.dim()
            # Slice off the padding at the end of the dimension
            slc = [slice(None)] * unpacked_int_data.dim()
            slc[dim] = slice(0, unpacked_int_data.shape[dim] - self._layout.pad_amount)
            unpacked_int_data = unpacked_int_data[tuple(slc)]

        return unpacked_int_data, self.scale, self.zero_point

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        # For slice/select/index which are structurally dependent, we can unpack, dispatch, and pack
        # Or delegate to plain implementation. Slicing packed dimensions directly is tricky if unaligned.
        # Just use the unpack-dispatch-pack formula for everything other than detach/clone/copy_/t/to.
        if func in [
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.select.int,
            torch.ops.aten.index.Tensor,
        ]:
            # Unpack first
            self = args[0]
            unpacked_int_data, scale, zero_point = self.get_plain()
            plain_impl = PlainAQTTensorImpl(
                unpacked_int_data, scale, zero_point, self._layout
            )
            # Dispatch to plain
            res_plain = PlainAQTTensorImpl.__torch_dispatch__(
                func, types, (plain_impl, *args[1:]), kwargs
            )
            # Re-pack if the operation returns a PlainAQTTensorImpl
            if isinstance(res_plain, PlainAQTTensorImpl):
                return cls.from_plain(
                    res_plain.int_data,
                    res_plain.scale,
                    res_plain.zero_point,
                    self._layout,
                )
            return res_plain

        if func is torch.ops.aten.t.default:
            self = args[0]
            unpacked_int_data, scale, zero_point = self.get_plain()
            # If pack_dim is -1, and we transpose, the packed dim changes.
            # In general, if we transpose, we could keep it plain or re-pack.
            plain_impl = PlainAQTTensorImpl(
                unpacked_int_data, scale, zero_point, self._layout
            )
            res_plain = PlainAQTTensorImpl.__torch_dispatch__(
                func, types, (plain_impl, *args[1:]), kwargs
            )
            if isinstance(res_plain, PlainAQTTensorImpl):
                # We do not repack if transposed since the dimension mappings shift.
                # Actually, best to return PlainAQTTensorImpl rather than crash.
                return res_plain
            return res_plain

        # Delegate simple properties to parent implementation (which uses _apply_fn_to_data)
        return super().__torch_dispatch__(func, types, args, kwargs)
