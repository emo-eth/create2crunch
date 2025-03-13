use metal::*;

// Safety wrapper for ComputeCommandEncoder to prevent context leaks
pub struct SafeEncoder<'a> {
    pub encoder: &'a ComputeCommandEncoderRef,
    ended: bool,
}

impl<'a> SafeEncoder<'a> {
    pub fn new(encoder: &'a ComputeCommandEncoderRef) -> Self {
        SafeEncoder {
            encoder,
            ended: false,
        }
    }

    pub fn end_encoding(&mut self) {
        if !self.ended {
            self.encoder.end_encoding();
            self.ended = true;
        }
    }
}

impl<'a> Drop for SafeEncoder<'a> {
    fn drop(&mut self) {
        self.end_encoding();
    }
}

// Safety wrapper for BlitCommandEncoder to prevent context leaks
pub struct SafeBlitEncoder<'a> {
    pub encoder: &'a BlitCommandEncoderRef,
    ended: bool,
}

impl<'a> SafeBlitEncoder<'a> {
    pub fn new(encoder: &'a BlitCommandEncoderRef) -> Self {
        SafeBlitEncoder {
            encoder,
            ended: false,
        }
    }

    pub fn end_encoding(&mut self) {
        if !self.ended {
            self.encoder.end_encoding();
            self.ended = true;
        }
    }
}

impl<'a> Drop for SafeBlitEncoder<'a> {
    fn drop(&mut self) {
        self.end_encoding();
    }
}
