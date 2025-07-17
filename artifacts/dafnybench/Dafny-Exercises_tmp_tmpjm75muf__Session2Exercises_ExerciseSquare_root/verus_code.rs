use vstd::prelude::*;

verus! {
    // Method 1: Linear search from 0 upward - Cost O(sqrt(n))
    fn mroot1(n: u32) -> (r: u32)
        requires n >= 0,
        ensures 
            r >= 0 && (r as int) * (r as int) <= n < ((r + 1) as int) * ((r + 1) as int),
    {
        let mut r: u32 = 0;
        while r < 65536 
            invariant 
                r >= 0 && (r as int) * (r as int) <= n,
            decreases 65536 - r,
        {
            // Check if we can increment r safely
            if r + 1 <= 65536 {
                // Use u64 to prevent overflow in the comparison
                let next_val = (r + 1) as u64;
                if next_val * next_val <= n as u64 {
                    r = r + 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        r
    }

    // Method 2: Linear search from n downward - Cost O(n)
    fn mroot2(n: u32) -> (r: u32)
        requires n >= 0,
        ensures 
            r >= 0 && (r as int) * (r as int) <= n < ((r + 1) as int) * ((r + 1) as int),
    {
        let mut r: u32 = if n <= 65536 { n } else { 65536 };
        
        while r > 0 && (r as u64) * (r as u64) > n as u64
            invariant 
                0 <= r,
            decreases r,
        {
            r = r - 1;
        }
        
        r
    }

    // Method 3: Binary search - Cost O(log n)
    fn mroot3(n: u32) -> (r: u32)
        requires 
            n >= 0,
            n < 0xFFFF_FFFE,
        ensures 
            r >= 0 && (r as int) * (r as int) <= n < ((r + 1) as int) * ((r + 1) as int),
    {
        let mut r: u32 = 0;
        let mut y: u32 = if n >= 65536 { 65536 } else { n + 1 };
        
        // Search in interval [r, y)
        while y > r + 1
            invariant 
                r >= 0 && (r as int) * (r as int) <= n,
                y >= r + 1,
            decreases (y as int) - (r as int),
        {
            let h: u32 = r + (y - r) / 2;
            let h_squared = (h as u64) * (h as u64);
            
            if h_squared <= n as u64 {
                r = h;
            } else {
                y = h;
            }
        }
        
        r
    }

    fn main() {
        // Test function to make the compilation work
    }
}