//! Minimal protobuf wire reader for the ONNX `ModelProto` subset the face-swap
//! models need. We do not link a protobuf crate: ONNX's schema is small and
//! stable, and we only ever READ (never write) a fixed subset of fields, so a
//! hand-rolled reader keeps the dependency footprint at zero and gives full
//! control over the f16/f32/i64 `raw_data` decode.
//!
//! Wire format (proto3): each field is `tag = (field_no << 3) | wire_type`
//! (varint), followed by the payload. Wire types: 0 varint, 1 64-bit, 2
//! length-delimited (the length is a varint prefix), 5 32-bit. Unknown fields
//! are skipped by wire type, so newer ONNX exports parse fine.
//!
//! Field numbers below are from `onnx.proto` (onnx/onnx repo). Only the fields
//! we consume are named; everything else is skipped.

use half::f16;

#[derive(Debug)]
pub enum ParseError {
    Truncated,
    BadWireType(u32),
    BadVarint,
    UnsupportedDtype(i32),
    /// `raw_data` byte length not a multiple of the element size.
    RaggedRawData {
        dtype: i32,
        bytes: usize,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Truncated => write!(f, "onnx: truncated protobuf"),
            ParseError::BadWireType(w) => write!(f, "onnx: bad wire type {w}"),
            ParseError::BadVarint => write!(f, "onnx: malformed varint"),
            ParseError::UnsupportedDtype(d) => write!(f, "onnx: unsupported tensor dtype {d}"),
            ParseError::RaggedRawData { dtype, bytes } => {
                write!(
                    f,
                    "onnx: raw_data of {bytes}B not aligned for dtype {dtype}"
                )
            }
        }
    }
}

impl std::error::Error for ParseError {}

// ONNX TensorProto.DataType (the subset these models use).
const DT_FLOAT: i32 = 1;
const DT_INT8: i32 = 3;
const DT_INT32: i32 = 6;
const DT_INT64: i32 = 7;
const DT_FLOAT16: i32 = 10;
const DT_DOUBLE: i32 = 11;

/// Decoded tensor payload. Numeric tensors collapse to either `F32` (weights,
/// floats) or `I64` (shapes, indices, axes) - the only two the executor + the
/// const-fold pass need. Other dtypes (e.g. int8 weights) widen into these.
#[derive(Clone, Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::I64(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// View as i64 (shape/index tensors). Panics on a float tensor - callers
    /// only invoke this on tensors a shape op produced.
    pub fn as_i64(&self) -> &[i64] {
        match self {
            TensorData::I64(v) => v,
            TensorData::F32(_) => panic!("onnx: expected i64 tensor, found f32"),
        }
    }
    /// View as f32 (weights / float constants), widening an i64 tensor if a
    /// graph feeds an int constant into a float op.
    pub fn to_f32(&self) -> std::borrow::Cow<'_, [f32]> {
        match self {
            TensorData::F32(v) => std::borrow::Cow::Borrowed(v),
            TensorData::I64(v) => std::borrow::Cow::Owned(v.iter().map(|&x| x as f32).collect()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<i64>,
    pub data: TensorData,
}

impl OnnxTensor {
    pub fn numel(&self) -> usize {
        self.dims.iter().product::<i64>().max(0) as usize
    }
}

#[derive(Clone, Debug)]
pub enum AttrValue {
    F(f32),
    I(i64),
    S(String),
    T(OnnxTensor),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct Attr {
    pub name: String,
    pub value: AttrValue,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<Attr>,
}

impl Node {
    pub fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attrs.iter().find(|a| a.name == name).map(|a| &a.value)
    }
    pub fn attr_i(&self, name: &str, default: i64) -> i64 {
        match self.attr(name) {
            Some(AttrValue::I(i)) => *i,
            _ => default,
        }
    }
    pub fn attr_f(&self, name: &str, default: f32) -> f32 {
        match self.attr(name) {
            Some(AttrValue::F(f)) => *f,
            _ => default,
        }
    }
    pub fn attr_ints(&self, name: &str) -> Option<&[i64]> {
        match self.attr(name) {
            Some(AttrValue::Ints(v)) => Some(v),
            _ => None,
        }
    }
    pub fn attr_s(&self, name: &str) -> Option<&str> {
        match self.attr(name) {
            Some(AttrValue::S(s)) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn attr_t(&self, name: &str) -> Option<&OnnxTensor> {
        match self.attr(name) {
            Some(AttrValue::T(t)) => Some(t),
            _ => None,
        }
    }
}

/// A declared graph input/output. `dims` carries `None` for dynamic axes
/// (symbolic dim_param or unset) so shape inference can bind them to the
/// concrete sizes the caller runs at.
#[derive(Clone, Debug)]
pub struct ValueInfo {
    pub name: String,
    pub dims: Vec<Option<i64>>,
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub initializers: Vec<OnnxTensor>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

// --- Wire reader ------------------------------------------------------------

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn eof(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64, ParseError> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            if self.pos >= self.buf.len() {
                return Err(ParseError::Truncated);
            }
            let byte = self.buf[self.pos];
            self.pos += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(ParseError::BadVarint);
            }
        }
    }

    /// Read a tag, returning `(field_number, wire_type)`.
    fn tag(&mut self) -> Result<(u32, u32), ParseError> {
        let t = self.varint()?;
        Ok(((t >> 3) as u32, (t & 0x7) as u32))
    }

    /// Read a length-delimited payload (wire type 2) and return its slice.
    fn bytes(&mut self) -> Result<&'a [u8], ParseError> {
        let len = self.varint()? as usize;
        if self.pos + len > self.buf.len() {
            return Err(ParseError::Truncated);
        }
        let s = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        Ok(s)
    }

    fn read_u32(&mut self) -> Result<u32, ParseError> {
        if self.pos + 4 > self.buf.len() {
            return Err(ParseError::Truncated);
        }
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64, ParseError> {
        if self.pos + 8 > self.buf.len() {
            return Err(ParseError::Truncated);
        }
        let v = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    /// Skip the payload of a field with the given wire type.
    fn skip(&mut self, wire: u32) -> Result<(), ParseError> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.read_u64()?;
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.read_u32()?;
            }
            w => return Err(ParseError::BadWireType(w)),
        }
        Ok(())
    }

    fn string(&mut self) -> Result<String, ParseError> {
        Ok(String::from_utf8_lossy(self.bytes()?).into_owned())
    }
}

/// Read a packed-repeated varint field (int64_data / int32_data when packed).
fn read_packed_varints(buf: &[u8]) -> Result<Vec<i64>, ParseError> {
    let mut r = Reader::new(buf);
    let mut out = Vec::new();
    while !r.eof() {
        out.push(r.varint()? as i64);
    }
    Ok(out)
}

/// Read a packed-repeated 32-bit-float field (float_data when packed).
fn read_packed_f32(buf: &[u8]) -> Result<Vec<f32>, ParseError> {
    if !buf.len().is_multiple_of(4) {
        return Err(ParseError::RaggedRawData {
            dtype: DT_FLOAT,
            bytes: buf.len(),
        });
    }
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Decode `raw_data` bytes per the tensor dtype into `TensorData`.
fn decode_raw(dtype: i32, raw: &[u8]) -> Result<TensorData, ParseError> {
    match dtype {
        DT_FLOAT => Ok(TensorData::F32(read_packed_f32(raw)?)),
        DT_DOUBLE => {
            if !raw.len().is_multiple_of(8) {
                return Err(ParseError::RaggedRawData {
                    dtype,
                    bytes: raw.len(),
                });
            }
            Ok(TensorData::F32(
                raw.chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect(),
            ))
        }
        DT_FLOAT16 => {
            if !raw.len().is_multiple_of(2) {
                return Err(ParseError::RaggedRawData {
                    dtype,
                    bytes: raw.len(),
                });
            }
            Ok(TensorData::F32(
                raw.chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
            ))
        }
        DT_INT64 => {
            if !raw.len().is_multiple_of(8) {
                return Err(ParseError::RaggedRawData {
                    dtype,
                    bytes: raw.len(),
                });
            }
            Ok(TensorData::I64(
                raw.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            ))
        }
        DT_INT32 => {
            if !raw.len().is_multiple_of(4) {
                return Err(ParseError::RaggedRawData {
                    dtype,
                    bytes: raw.len(),
                });
            }
            Ok(TensorData::I64(
                raw.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            ))
        }
        DT_INT8 => Ok(TensorData::I64(
            raw.iter().map(|&b| b as i8 as i64).collect(),
        )),
        d => Err(ParseError::UnsupportedDtype(d)),
    }
}

/// Parse a `TensorProto`.
fn parse_tensor(buf: &[u8]) -> Result<OnnxTensor, ParseError> {
    let mut r = Reader::new(buf);
    let mut dims: Vec<i64> = Vec::new();
    let mut dtype: i32 = DT_FLOAT;
    let mut name = String::new();
    let mut raw: Option<&[u8]> = None;
    let mut float_data: Option<Vec<f32>> = None;
    let mut int64_data: Option<Vec<i64>> = None;
    let mut int32_data: Option<Vec<i64>> = None;
    while !r.eof() {
        let (field, wire) = r.tag()?;
        match (field, wire) {
            // dims: repeated int64. Packed (wire 2) or one-per-tag (wire 0).
            (1, 2) => dims.extend(read_packed_varints(r.bytes()?)?),
            (1, 0) => dims.push(r.varint()? as i64),
            (2, 0) => dtype = r.varint()? as i32,
            (4, 2) => float_data = Some(read_packed_f32(r.bytes()?)?),
            (5, 2) => int32_data = Some(read_packed_varints(r.bytes()?)?),
            (7, 2) => int64_data = Some(read_packed_varints(r.bytes()?)?),
            (8, 2) => name = r.string()?,
            (9, 2) => raw = Some(r.bytes()?),
            _ => r.skip(wire)?,
        }
    }
    let data = if let Some(raw) = raw {
        decode_raw(dtype, raw)?
    } else if let Some(v) = float_data {
        TensorData::F32(v)
    } else if let Some(v) = int64_data {
        TensorData::I64(v)
    } else if let Some(v) = int32_data {
        TensorData::I64(v)
    } else {
        // Empty tensor (e.g. an optional input placeholder).
        match dtype {
            DT_INT64 | DT_INT32 | DT_INT8 => TensorData::I64(Vec::new()),
            _ => TensorData::F32(Vec::new()),
        }
    };
    Ok(OnnxTensor { name, dims, data })
}

/// Parse an `AttributeProto`.
fn parse_attr(buf: &[u8]) -> Result<Attr, ParseError> {
    let mut r = Reader::new(buf);
    let mut name = String::new();
    let mut atype: i64 = 0;
    let mut f: f32 = 0.0;
    let mut i: i64 = 0;
    let mut s: Option<String> = None;
    let mut t: Option<OnnxTensor> = None;
    let mut floats: Vec<f32> = Vec::new();
    let mut ints: Vec<i64> = Vec::new();
    let mut strings: Vec<String> = Vec::new();
    while !r.eof() {
        let (field, wire) = r.tag()?;
        match (field, wire) {
            (1, 2) => name = r.string()?,
            (2, 5) => f = f32::from_bits(r.read_u32()?),
            (3, 0) => i = r.varint()? as i64,
            (4, 2) => s = Some(r.string()?),
            (5, 2) => t = Some(parse_tensor(r.bytes()?)?),
            // floats: packed (2) or single (5).
            (7, 2) => floats.extend(read_packed_f32(r.bytes()?)?),
            (7, 5) => floats.push(f32::from_bits(r.read_u32()?)),
            // ints: packed (2) or single (0).
            (8, 2) => ints.extend(read_packed_varints(r.bytes()?)?),
            (8, 0) => ints.push(r.varint()? as i64),
            (9, 2) => strings.push(r.string()?),
            (20, 0) => atype = r.varint()? as i64,
            _ => r.skip(wire)?,
        }
    }
    // AttributeType: FLOAT=1 INT=2 STRING=3 TENSOR=4 FLOATS=6 INTS=7 STRINGS=8.
    let value = match atype {
        1 => AttrValue::F(f),
        2 => AttrValue::I(i),
        3 => AttrValue::S(s.unwrap_or_default()),
        4 => AttrValue::T(t.ok_or(ParseError::Truncated)?),
        6 => AttrValue::Floats(floats),
        7 => AttrValue::Ints(ints),
        8 => AttrValue::Strings(strings),
        // Fall back by which payload is populated (older exports omit `type`).
        _ if t.is_some() => AttrValue::T(t.unwrap()),
        _ if !ints.is_empty() => AttrValue::Ints(ints),
        _ if !floats.is_empty() => AttrValue::Floats(floats),
        _ if !strings.is_empty() => AttrValue::Strings(strings),
        _ if s.is_some() => AttrValue::S(s.unwrap()),
        _ => AttrValue::I(i),
    };
    Ok(Attr { name, value })
}

/// Parse a `NodeProto`.
fn parse_node(buf: &[u8]) -> Result<Node, ParseError> {
    let mut r = Reader::new(buf);
    let mut node = Node {
        op_type: String::new(),
        name: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Vec::new(),
    };
    while !r.eof() {
        let (field, wire) = r.tag()?;
        match (field, wire) {
            (1, 2) => node.inputs.push(r.string()?),
            (2, 2) => node.outputs.push(r.string()?),
            (3, 2) => node.name = r.string()?,
            (4, 2) => node.op_type = r.string()?,
            (5, 2) => node.attrs.push(parse_attr(r.bytes()?)?),
            _ => r.skip(wire)?,
        }
    }
    Ok(node)
}

/// Parse a `TensorShapeProto.Dimension` -> `Some(value)` for a fixed axis,
/// `None` for a symbolic / unset one.
fn parse_dim(buf: &[u8]) -> Result<Option<i64>, ParseError> {
    let mut r = Reader::new(buf);
    let mut val: Option<i64> = None;
    while !r.eof() {
        let (field, wire) = r.tag()?;
        match (field, wire) {
            (1, 0) => val = Some(r.varint()? as i64),
            // dim_param (symbolic) -> stays None.
            _ => r.skip(wire)?,
        }
    }
    Ok(val)
}

/// Parse a `ValueInfoProto` (drills through TypeProto -> Tensor -> Shape).
fn parse_value_info(buf: &[u8]) -> Result<ValueInfo, ParseError> {
    let mut r = Reader::new(buf);
    let mut name = String::new();
    let mut dims: Vec<Option<i64>> = Vec::new();
    while !r.eof() {
        let (field, wire) = r.tag()?;
        match (field, wire) {
            (1, 2) => name = r.string()?,
            (2, 2) => dims = parse_type(r.bytes()?)?,
            _ => r.skip(wire)?,
        }
    }
    Ok(ValueInfo { name, dims })
}

/// TypeProto -> the shape dims (field 1: tensor_type).
fn parse_type(buf: &[u8]) -> Result<Vec<Option<i64>>, ParseError> {
    let mut r = Reader::new(buf);
    let mut dims = Vec::new();
    while !r.eof() {
        let (field, wire) = r.tag()?;
        match (field, wire) {
            (1, 2) => dims = parse_tensor_type(r.bytes()?)?,
            _ => r.skip(wire)?,
        }
    }
    Ok(dims)
}

/// TypeProto.Tensor -> shape dims (field 2: shape).
fn parse_tensor_type(buf: &[u8]) -> Result<Vec<Option<i64>>, ParseError> {
    let mut r = Reader::new(buf);
    let mut dims = Vec::new();
    while !r.eof() {
        let (field, wire) = r.tag()?;
        match (field, wire) {
            (2, 2) => dims = parse_shape(r.bytes()?)?,
            _ => r.skip(wire)?,
        }
    }
    Ok(dims)
}

/// TensorShapeProto -> repeated Dimension (field 1).
fn parse_shape(buf: &[u8]) -> Result<Vec<Option<i64>>, ParseError> {
    let mut r = Reader::new(buf);
    let mut dims = Vec::new();
    while !r.eof() {
        let (field, wire) = r.tag()?;
        match (field, wire) {
            (1, 2) => dims.push(parse_dim(r.bytes()?)?),
            _ => r.skip(wire)?,
        }
    }
    Ok(dims)
}

/// Parse a `GraphProto`.
fn parse_graph(buf: &[u8]) -> Result<Graph, ParseError> {
    let mut r = Reader::new(buf);
    let mut graph = Graph {
        nodes: Vec::new(),
        initializers: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    while !r.eof() {
        let (field, wire) = r.tag()?;
        match (field, wire) {
            (1, 2) => graph.nodes.push(parse_node(r.bytes()?)?),
            (5, 2) => graph.initializers.push(parse_tensor(r.bytes()?)?),
            (11, 2) => graph.inputs.push(parse_value_info(r.bytes()?)?),
            (12, 2) => graph.outputs.push(parse_value_info(r.bytes()?)?),
            _ => r.skip(wire)?,
        }
    }
    Ok(graph)
}

/// Parse an ONNX `ModelProto` and return its graph (field 7).
pub fn parse_model(bytes: &[u8]) -> Result<Graph, ParseError> {
    let mut r = Reader::new(bytes);
    let mut graph: Option<Graph> = None;
    while !r.eof() {
        let (field, wire) = r.tag()?;
        match (field, wire) {
            (7, 2) => graph = Some(parse_graph(r.bytes()?)?),
            _ => r.skip(wire)?,
        }
    }
    graph.ok_or(ParseError::Truncated)
}
