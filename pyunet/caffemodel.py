"""Read the weights out of an old-format Caffe .caffemodel file.

A .caffemodel is a serialised protobuf `NetParameter`. Rather than compiling
caffe.proto, this decodes the few fields we need straight from the protobuf
wire format, so the only dependency is numpy.

Field numbers come from u-net-release/caffe-unet-src/src/caffe/proto/caffe.proto:
    NetParameter.layers = 2   (repeated LayerParameter, the old "V1" style)
    LayerParameter.name = 4,  LayerParameter.blobs = 6
    BlobProto: num = 1, channels = 2, height = 3, width = 4, data = 5 (packed float)
"""

from collections import OrderedDict

import numpy as np

_NET_LAYERS = 2
_LAYER_NAME, _LAYER_BLOBS = 4, 6
_BLOB_NUM, _BLOB_CHANNELS, _BLOB_HEIGHT, _BLOB_WIDTH, _BLOB_DATA = 1, 2, 3, 4, 5

_VARINT, _FIXED64, _LENGTH_DELIMITED, _FIXED32 = 0, 1, 2, 5


def _read_varint(buf, pos):
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf):
    """Yield (field_number, wire_type, value) for each field in a message."""
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == _VARINT:
            value, pos = _read_varint(buf, pos)
        elif wire == _FIXED64:
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire == _LENGTH_DELIMITED:
            length, pos = _read_varint(buf, pos)
            value, pos = buf[pos:pos + length], pos + length
        elif wire == _FIXED32:
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield field, wire, value


def _parse_blob(buf):
    dims = {_BLOB_NUM: 1, _BLOB_CHANNELS: 1, _BLOB_HEIGHT: 1, _BLOB_WIDTH: 1}
    chunks = []
    for field, wire, value in _iter_fields(buf):
        if field in dims:
            dims[field] = value
        elif field == _BLOB_DATA:
            # Packed floats arrive as one length-delimited run; unpacked ones as
            # one fixed32 per value. Both are little-endian float32.
            chunks.append(np.frombuffer(value, dtype="<f4"))
    data = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    shape = tuple(dims[f] for f in (_BLOB_NUM, _BLOB_CHANNELS, _BLOB_HEIGHT, _BLOB_WIDTH))
    if data.size != np.prod(shape):
        raise ValueError(f"Blob has {data.size} values but shape {shape}")
    return data.astype(np.float32).reshape(shape)


def read_caffemodel(path):
    """Return an OrderedDict {layer_name: [blob arrays]} in file order.

    Each blob keeps Caffe's 4-D (num, channels, height, width) shape. Layers
    with no parameters (ReLU, pooling, crop, ...) map to an empty list.
    """
    with open(path, "rb") as f:
        buf = memoryview(f.read())

    layers = OrderedDict()
    for field, wire, value in _iter_fields(buf):
        if field != _NET_LAYERS or wire != _LENGTH_DELIMITED:
            continue
        name, blobs = None, []
        for lfield, lwire, lvalue in _iter_fields(value):
            if lfield == _LAYER_NAME:
                name = bytes(lvalue).decode("utf-8")
            elif lfield == _LAYER_BLOBS:
                blobs.append(_parse_blob(lvalue))
        layers[name] = blobs

    if not layers:
        raise ValueError(
            f"No V1 'layers' entries found in {path}; this reader only handles "
            "the old Caffe format used by u-net-release."
        )
    return layers
