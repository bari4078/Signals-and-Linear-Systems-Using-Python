# CSE220 – DFT/FFT Online Test Prep Guide

This guide covers **only the past online tests**, not the full assignment. It assumes you've already built your FFT/DFT engine in the assignment (`transforms.py`), so here we just **use it as a ready-made tool** — you don't need to know how it works internally, only how to call it.

---

## 0. Your Toolkit (treat this as a black box)

You already wrote this. Just import and use it:

```python
from transforms import FFTTransformer, next_power_of_two

engine = FFTTransformer()
X = engine.transform(x)     # time domain -> frequency domain (complex array)
x = engine.inverse(X)       # frequency domain -> time domain (complex array)
```

**Rules to remember:**
- `x` must be a NumPy array (real or complex numbers are both fine).
- The **length of `x` must be a power of two** (2, 4, 8, 16, 32, ... 256, 512, ...). If it isn't, pad it with zeros first using `next_power_of_two(n)`.
- `engine.inverse(X)` gives back **complex** numbers. If you know your answer should be real (which is almost always the case in these problems), take `.real` at the end.

---

## 1. Concepts You Need (no derivations, just the "what" and "how")

### 1.1 The one formula that solves almost every problem here

```
IFFT( FFT(a) * FFT(b) )  =  circular convolution of a and b
```

Multiplying two spectra point-by-point, then inverse-transforming, gives you the **circular convolution** of the two original signals. This single idea is behind polynomial multiplication, big-integer multiplication, and image blurring.

### 1.2 Where convolution shows up
| Real-world task | Is secretly... |
|---|---|
| Multiplying two polynomials | Linear convolution of their coefficient arrays |
| Multiplying two big integers | Linear convolution of their digit arrays (+ carrying) |
| Encrypting/blurring an image row with a "key" | Circular convolution of the row with the key |

### 1.3 Turning circular convolution into linear convolution (zero-padding)

FFT-based multiplication always gives you the **circular** version, but polynomial/integer multiplication needs the **linear** version. The fix is padding:

- The true linear-convolution result has length `len(a) + len(b) - 1`.
- **Zero-pad both arrays to at least that length** (then further up to the next power of two, since `FFTTransformer` requires that) before transforming.
- After you inverse-transform, **only keep the first `len(a) + len(b) - 1` entries** — anything after that is wraparound junk from the padding.

If you *skip* this padding, the high-order terms wrap around and corrupt your low-order terms (aliasing).

### 1.4 Cross-correlation via FFT (for finding shifts)

```
corr = IFFT( FFT(a) * conj(FFT(b)) )
```

- `conj(...)` = complex conjugate (`np.conj(x)`).
- The **index of the largest value** in `corr` (`np.argmax`) tells you how far `b` is shifted relative to `a`.
- To undo a circular shift, roll the signal backward by that amount: `np.roll(b, -shift)`.

### 1.5 Carry propagation (needed after any digit/limb convolution)

Raw convolution output can be bigger than a single digit (e.g. 15, 22, 13 instead of 0–9). Fix it by sweeping left to right:

```python
carry = 0
for i in range(len(raw)):
    raw[i] += carry
    digit = raw[i] % 10
    carry  = raw[i] // 10
    raw[i] = digit
# then keep appending carry % 10 while carry > 0
```

---

## 2. Python / NumPy Syntax Refresher (only what you'll actually type)

| Syntax | What it does |
|---|---|
| `1j` | The imaginary unit `i`. `3 + 4j` is a complex number. |
| `x.real`, `x.imag` | Pulls out the real or imaginary part of a complex number/array. |
| `np.conj(x)` | Complex conjugate (flips the sign of the imaginary part). Needed for cross-correlation. |
| `np.pad(arr, (0, k))` | Pads `arr` with `k` zeros at the end. |
| `np.round(x).astype(int)` | Rounds floats to the nearest integer and converts the array to integer type — needed because IFFT output has tiny floating-point noise. |
| `arr[::-1]` | Reverses an array or string. |
| `s[::-1]` on a string | Same idea — reverses a string, e.g. turning "123" into "321" (handy for LSD-first digit arrays). |
| `np.roll(arr, k)` | Circularly shifts an array by `k` positions (positive = shifts right/down). |
| `np.argmax(arr)` / `np.argmin(arr)` | Index of the largest/smallest element. |
| `divmod(a, b)` | Returns `(a // b, a % b)` together — useful for carry propagation. |
| `zip(a, b)` | Pairs up two lists element-by-element: `zip([1,2],[3,4])` → `(1,3), (2,4)`. |
| `[expr for x, y in zip(a, b)]` | A list comprehension — builds a new list by applying `expr` to each pair. |
| `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | Converts a color image (loaded by OpenCV) to grayscale — a plain 2D array of brightness values. |
| `np.clip(x, 0, 255)` | Forces every value into the range 0–255 (valid pixel range) before saving an image. |

---

## 3. The Problems

Each problem below is restated briefly in terms of the toolkit from Section 0, followed by a short, self-contained solution.

---

### Problem 1 — Weighted Polynomial Multiplication

**Problem statement:** You're given two polynomials `P`, `Q` (coefficients in descending power order) and a weight array `W` the same length as `P`. Compute the product polynomial `R`, but first scale each coefficient of `P` by the matching weight in `W`. You must compute this using FFT/IFFT-based circular convolution (not direct/naive multiplication).

**Key idea:** This is just linear convolution with one extra step (weighting `P` first). Weight → zero-pad → FFT → multiply spectra → IFFT → crop to linear length → reverse back to descending order.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def weighted_polynomial_multiply(P, Q, W):
    engine = FFTTransformer()

    Pw = [p * w for p, w in zip(P, W)]        # weight each coefficient of P

    lin_len = len(Pw) + len(Q) - 1             # true linear-convolution length
    N = next_power_of_two(lin_len)             # FFT needs a power-of-two length

    Pw_pad = np.pad(Pw, (0, N - len(Pw)))
    Q_pad  = np.pad(Q,  (0, N - len(Q)))

    spectrum = engine.transform(Pw_pad) * engine.transform(Q_pad)
    conv = engine.inverse(spectrum).real
    conv = np.round(conv).astype(int)[:lin_len]   # crop off the padding junk

    return conv[::-1].tolist()                 # back to descending-power order

# Example:
P = [1, 3, 2, 6, 7]
Q = [4, 1]
W = [3, 2, 1, 5, 6]
print("Result:", weighted_polynomial_multiply(P, Q, W))
# Result: [42, 198, 122, 14, 27, 12]
```

---

### Problem 2 — Big-Integer Multiplication via FFT

**Problem statement:** You're given two large non-negative integers as decimal strings. Multiply them using FFT/IFFT-based circular convolution over their digit arrays (not Python's built-in big-integer multiplication), then propagate carries to get the correct final digit sequence.

**Key idea:** Convert each number to an LSD-first digit array → zero-pad → FFT-multiply → IFFT → carry-propagate.

```python
import numpy as np
from transforms import FFTTransformer, next_power_of_two

def multiply_big_numbers(a_str, b_str):
    engine = FFTTransformer()

    A = [int(d) for d in a_str[::-1]]   # LSD-first digit array
    B = [int(d) for d in b_str[::-1]]

    lin_len = len(A) + len(B) - 1
    N = next_power_of_two(lin_len)

    A_pad = np.pad(A, (0, N - len(A)))
    B_pad = np.pad(B, (0, N - len(B)))

    conv = engine.inverse(engine.transform(A_pad) * engine.transform(B_pad)).real
    conv = np.round(conv).astype(int)[:lin_len]

    # carry propagation
    result, carry = [], 0
    for c in conv:
        c += carry
        result.append(c % 10)
        carry = c // 10
    while carry:
        result.append(carry % 10)
        carry //= 10

    while len(result) > 1 and result[-1] == 0:   # strip leading (MSD-side) zeros
        result.pop()

    return result   # LSD to MSD, as the task requires

print(multiply_big_numbers("123", "45"))   # [5, 3, 5, 5]  -> 5535
print(multiply_big_numbers("999", "99"))   # [1, 0, 9, 8, 9] -> 98901
```

This exact function also handles the "huge number" version of this problem (e.g. `65767879797907 × 765454532435435345`) — no changes needed, it just gets a bigger `N`.

---

### Problem 3 — Rolling-Shutter Row-Shift Reconstruction

**Problem statement:** A camera's rolling shutter caused every row of an image to be circularly shifted right by a different, unknown amount (no vertical shift). Given the original and the distorted image, detect each row's shift using DFT-based cross-correlation and reverse it to reconstruct the original.

**Key idea:** For each row, cross-correlate the original row against the shifted row; the peak position is the shift amount. Undo it with `np.roll`.

```python
import cv2
import numpy as np
from transforms import FFTTransformer

def find_shift(ref_row, shifted_row, engine):
    corr = engine.inverse(engine.transform(ref_row) * np.conj(engine.transform(shifted_row))).real
    return int(np.argmax(corr))
    # Note: this works directly (no padding) because the row length in these
    # tasks is already a power of two (e.g. 256 or 512 pixels wide).

def reconstruct_image_using_fft(original_path, shifted_path, output_path):
    original_img = cv2.imread(original_path)
    shifted_img  = cv2.imread(shifted_path)

    orig_gray  = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY).astype(float)
    shift_gray = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2GRAY).astype(float)

    engine = FFTTransformer()
    reconstructed = np.zeros_like(shift_gray)

    for r in range(shift_gray.shape[0]):
        shift = find_shift(orig_gray[r], shift_gray[r], engine)
        reconstructed[r] = np.roll(shift_gray[r], -shift)

    cv2.imwrite(output_path, reconstructed.astype(np.uint8))
```

---

### Problem 4 — BUET Logo Decryption (circular deconvolution)

**Problem statement:** Every row of an image (except one "key row") was circularly convolved with that key row. The key row is identifiable because its pixel values are much smaller than the encrypted rows'. Find the key row, then reverse the convolution on every other row to recover the original image.

**Key idea:** Circular convolution in the time domain = multiplication in the frequency domain. So to *undo* it, **divide** spectra instead of multiplying: `original = IFFT( FFT(encrypted) / FFT(key) )`.

```python
import numpy as np
from PIL import Image
from transforms import FFTTransformer

def decrypt_buet_logo(encrypted_path, output_path):
    img = np.array(Image.open(encrypted_path).convert("L")).astype(float)
    engine = FFTTransformer()

    col = 0                                     # any column works, per the hint
    key_row_idx = int(np.argmin(img[:, col]))    # key row = the "humble" (smallest) row
    key_row = img[key_row_idx]

    Key_F = engine.transform(key_row)
    decrypted = np.zeros_like(img)
    decrypted[key_row_idx] = key_row             # key row was never encrypted

    for r in range(img.shape[0]):
        if r == key_row_idx:
            continue
        Row_F = engine.transform(img[r])
        orig_F = Row_F / Key_F                   # deconvolution = division in frequency domain
        decrypted[r] = engine.inverse(orig_F).real

    decrypted = np.clip(np.round(decrypted), 0, 255).astype(np.uint8)
    Image.fromarray(decrypted).save(output_path)
```

---

### Problem 5 — 2D Shift Detection & Correction

**Problem statement:** A satellite image was shifted both horizontally and vertically compared to the original (no rotation/scaling). Using DFT-based cross-correlation on one row and one column, detect the horizontal and vertical shift amounts, then reverse both to realign the image.

**Key idea:** Same cross-correlation trick as Problem 3, applied once along a row (→ horizontal shift) and once along a column (→ vertical shift). Pick a row/column that actually has variation in it — a flat/blank one gives a meaningless peak.

```python
import numpy as np
from transforms import FFTTransformer

def find_1d_shift(a, b, engine):
    corr = engine.inverse(engine.transform(a) * np.conj(engine.transform(b))).real
    return int(np.argmax(corr))

def realign_images(original, shifted, row_idx, col_idx):
    engine = FFTTransformer()

    h_shift = find_1d_shift(original[row_idx, :], shifted[row_idx, :], engine)
    v_shift = find_1d_shift(original[:, col_idx], shifted[:, col_idx], engine)

    realigned = np.roll(shifted, shift=(-v_shift, -h_shift), axis=(0, 1))
    return realigned
```

---

## 4. Cheat Sheet — Formulas at a Glance

| Goal | Formula |
|---|---|
| Circular convolution | `IFFT( FFT(a) * FFT(b) )` |
| Linear convolution (via circular) | Zero-pad both to `len(a)+len(b)-1` (then to a power of two) → circular convolve → keep only the first `len(a)+len(b)-1` values |
| Cross-correlation (shift detection) | `IFFT( FFT(a) * conj(FFT(b)) )`, shift = `argmax` of the result |
| Undo a circular shift by `s` | `np.roll(arr, -s)` |
| Deconvolution (undo a convolution) | `IFFT( FFT(encrypted) / FFT(key) )` |
| Carry propagation | `digit = value % base; carry = value // base` |

---

## 5. Quick Self-Check Before the Test

- [ ] I can zero-pad two arrays to the right length for **linear** convolution.
- [ ] I know cross-correlation uses `conj()` and convolution doesn't.
- [ ] I remember to `.real` and round after every `inverse()` call.
- [ ] I know `np.roll(arr, -shift)` undoes a shift found via correlation peak.
- [ ] I can write carry propagation from memory without looking it up.
