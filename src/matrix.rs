use num_traits::{One, Zero, one, zero};
use std::array;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<T, const N: usize> {
    data: [T; N],
}

// An R x C matrix, stored in row-major order
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    data: [Vector<T, C>; R],
}

// Indexing
impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C> {
    type Output = Vector<T, C>; // The output type is a Vector

    fn index(&self, row: usize) -> &Self::Output {
        &self.data[row]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<usize> for Matrix<T, R, C> {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        &mut self.data[row]
    }
}

// AsRef
impl<T, const N: usize> AsRef<[T]> for Vector<T, N> {
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<T, const N: usize> AsMut<[T]> for Vector<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

// Arithmetic operations
// Vector addition (vector1 + vector2)
impl<T, const N: usize> Add for Vector<T, N>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Vector {
            data: array::from_fn(|i| self.data[i] + rhs.data[i]),
        }
    }
}

// Vector addition in-place (vector1 += vector2)
impl<T, const N: usize> AddAssign for Vector<T, N>
where
    T: AddAssign + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.data[i] += rhs.data[i];
        }
    }
}

// Matrix addition (matrix1 + matrix2)
impl<T, const R: usize, const C: usize> Add for Matrix<T, R, C>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Matrix {
            // This works by calling the `Add` implementation for Vector
            data: array::from_fn(|i| self.data[i] + rhs.data[i]),
        }
    }
}

// Matrix addition in-place (matrix1 += matrix2)
impl<T, const R: usize, const C: usize> AddAssign for Matrix<T, R, C>
where
    Vector<T, C>: AddAssign + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..R {
            self.data[i] += rhs.data[i]; // Calls Vector's AddAssign
        }
    }
}

// Matrix multiplication
impl<T, const M: usize, const N: usize, const K: usize> Mul<Matrix<T, { N }, { K }>>
    for Matrix<T, { M }, { N }>
where
    T: Zero + AddAssign + Mul<Output = T> + Copy,
{
    type Output = Matrix<T, { M }, { K }>;

    fn mul(self, rhs: Matrix<T, { N }, { K }>) -> Self::Output {
        Matrix {
            data: array::from_fn(|m| {
                Vector::from(array::from_fn(|k| {
                    let mut sum = zero();
                    for n in 0..N {
                        sum += self[m][n] * rhs[n][k];
                    }
                    sum
                }))
            }),
        }
    }
}

// Scalar multiplication for vectors (vector * scalar)
impl<T, const N: usize> Mul<T> for Vector<T, N>
where
    T: Mul<Output = T> + Copy + Default,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self::Output {
        Vector {
            data: array::from_fn(|i| self.data[i] * scalar),
        }
    }
}

// Scalar multiplication for vectors, in-place (matrix *= scalar)
impl<T, const N: usize> MulAssign<T> for Vector<T, N>
where
    T: MulAssign + Copy,
{
    fn mul_assign(&mut self, scalar: T) {
        for val in self.data.iter_mut() {
            *val *= scalar;
        }
    }
}

// Scalar multiplication for matrices (matrix * scalar)
impl<T, const R: usize, const C: usize> Mul<T> for Matrix<T, R, C>
where
    T: Mul<Output = T> + Copy + Default,
    Vector<T, C>: Mul<T, Output = Vector<T, C>> + Default + Copy,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self::Output {
        Matrix {
            data: array::from_fn(|i| self.data[i] * scalar),
        }
    }
}

// Scalar multiplication for matrices, in-place (matrix *= scalar)
impl<T, const R: usize, const C: usize> MulAssign<T> for Matrix<T, R, C>
where
    T: MulAssign + Copy,
    Vector<T, C>: MulAssign<T>, // We just need the Vector impl
{
    fn mul_assign(&mut self, scalar: T) {
        // Just scale each row
        for row in self.data.iter_mut() {
            *row *= scalar;
        }
    }
}

// Vector negation (-vector)
impl<T, const N: usize> Neg for Vector<T, N>
where
    T: Neg<Output = T> + Copy,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vector {
            data: array::from_fn(|i| -self.data[i]),
        }
    }
}

// Matrix negation (-matrix)
impl<T, const R: usize, const C: usize> Neg for Matrix<T, R, C>
where
    // Vector<T, C> must implement Neg
    Vector<T, C>: Neg<Output = Vector<T, C>> + Copy,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Matrix {
            data: array::from_fn(|i| -self.data[i]), // Calls Vector's Neg
        }
    }
}

// Vector subtraction (vector1 - vector2)
impl<T, const N: usize> Sub for Vector<T, N>
where
    // T must support subtraction and be copyable
    T: Sub<Output = T> + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector {
            // Create a new array by subtracting elements at each index
            data: array::from_fn(|i| self.data[i] - rhs.data[i]),
        }
    }
}

// Vector subtraction in-place (vector1 -= vector2)
impl<T, const N: usize> SubAssign for Vector<T, N>
where
    // T must support in-place subtraction
    T: SubAssign + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.data[i] -= rhs.data[i];
        }
    }
}

// Vector subtraction (matrix1 - matrix2)
impl<T, const R: usize, const C: usize> Sub for Matrix<T, R, C>
where
    // T's bounds must satisfy the Vector<T, C> subtraction
    T: Sub<Output = T> + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Matrix {
            // Create a new array of row vectors
            // This works by calling the `Sub` implementation for Vector
            data: array::from_fn(|i| self.data[i] - rhs.data[i]),
        }
    }
}

// Matrix subtraction in-place (matrix1 -= matrix2)
impl<T, const R: usize, const C: usize> SubAssign for Matrix<T, R, C>
where
    // Vector<T, C> must support in-place subtraction
    Vector<T, C>: SubAssign + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..R {
            self.data[i] -= rhs.data[i]; // Calls Vector's SubAssign
        }
    }
}

// Zero
impl<T: Copy + Zero, const N: usize> Zero for Vector<T, N> {
    fn zero() -> Self {
        Self { data: [zero(); N] }
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.is_zero())
    }
}

impl<T: Copy + Zero, const R: usize, const C: usize> Zero for Matrix<T, R, C> {
    fn zero() -> Self {
        Self {
            data: [Vector::zero(); R],
        }
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|row| row.is_zero())
    }
}

// One
impl<T, const N: usize> One for Matrix<T, N, N>
where
    T: Zero + One + Copy,
    Self: Mul<Self, Output = Self> + Zero,
{
    fn one() -> Self {
        Self::identity()
    }
}

// Defaults
impl<T: Copy + Default, const N: usize> Default for Vector<T, N> {
    fn default() -> Self {
        Self {
            data: [T::default(); N],
        }
    }
}

impl<T: Copy + Default, const R: usize, const C: usize> Default for Matrix<T, R, C> {
    fn default() -> Self {
        Self {
            data: [Vector::default(); R],
        }
    }
}

// From
impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T, R, C> {
    fn from(array: [[T; C]; R]) -> Self {
        let data = array.map(Vector::from);
        Self { data }
    }
}

impl<T, const N: usize> From<Vector<T, N>> for Matrix<T, 1, N>
where
    T: Copy,
    Vector<T, N>: Copy,
{
    fn from(vector: Vector<T, N>) -> Self {
        Matrix { data: [vector] }
    }
}

// Vector methods
impl<T, const N: usize> Vector<T, N>
where
    T: Copy,
{
    /// Transposes this row-like vector into a Nx1 column matrix.
    pub fn transpose(self) -> Matrix<T, N, 1>
    where
        T: Default,
    {
        let mut new = Matrix::<T, N, 1>::default();

        for i in 0..N {
            new[i][0] = self[i];
        }
        new
    }
    /// Calculates the dot product of two vectors.
    pub fn dot(self, rhs: Self) -> T
    where
        T: Zero + AddAssign + Mul<Output = T>,
    {
        let mut sum = zero();
        for i in 0..N {
            sum += self[i] * rhs[i];
        }
        sum
    }
}

// Matrix methods
impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Copy + Default,
{
    /// Matrix transposition
    pub fn transpose(self) -> Matrix<T, C, R> {
        let mut new = Matrix::default();

        for r in 0..R {
            for c in 0..C {
                new[c][r] = self[r][c];
            }
        }
        new
    }
}

// Methods for square matrices
impl<T, const N: usize> Matrix<T, N, N>
where
    T: Copy,
{
    /// Generate an identity matrix
    pub fn identity() -> Self
    where
        T: Zero + One,
    {
        Self {
            data: array::from_fn(|i| {
                let mut data = [zero(); N];
                data[i] = one();
                Vector::from(data)
            }),
        }
    }

    /// In-place matrix tramsposition
    pub fn transpose_mut(&mut self) {
        for r in 0..N {
            for c in (r + 1)..N {
                let tmp = self[r][c];
                self[r][c] = self[c][r];
                self[c][r] = tmp;
            }
        }
    }

    pub fn trace(&self) -> T
    where
        T: Zero + AddAssign,
    {
        let mut sum = zero();
        for i in 0..N {
            sum += self[i][i];
        }
        sum
    }
}

impl<T> Matrix<T, 3, 1>
where
    T: Mul<Output = T> + Sub<Output = T> + Copy,
{
    pub fn cross(&self, other: &Matrix<T, 3, 1>) -> Matrix<T, 3, 1> {
        let a1 = self[0][0];
        let a2 = self[1][0];
        let a3 = self[2][0];

        let b1 = other[0][0];
        let b2 = other[1][0];
        let b3 = other[2][0];

        let c1 = a2 * b3 - a3 * b2;
        let c2 = a3 * b1 - a1 * b3;
        let c3 = a1 * b2 - a2 * b1;

        Matrix::from([[c1], [c2], [c3]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_indexing() {
        let mut v: Vector<i32, 3> = Vector::default();
        assert_eq!(v[0], 0);

        v[1] = 10;
        assert_eq!(v[1], 10);

        // Test the underlying data
        assert_eq!(v.data, [0, 10, 0]);
    }

    #[test]
    fn test_vector_add_and_add_assign() {
        let mut v1 = Vector::from([1, 2, 3]);
        let v2 = Vector::from([10, 20, 30]);

        // Test Add (out-of-place)
        let v3 = v1 + v2;
        let expected = Vector::from([11, 22, 33]);
        assert_eq!(v3, expected);

        // Test that original is unchanged
        assert_eq!(v1, Vector::from([1, 2, 3]));

        // Test AddAssign (in-place)
        v1 += v2;
        assert_eq!(v1, expected);
    }

    #[test]
    fn test_vector_sub_and_sub_assign() {
        let mut v1 = Vector::from([10, 20, 30]);
        let v2 = Vector::from([1, 2, 3]);

        // Test Sub (out-of-place)
        let v3 = v1 - v2;
        let expected = Vector::from([9, 18, 27]);
        assert_eq!(v3, expected);

        // Test that original is unchanged
        assert_eq!(v1, Vector::from([10, 20, 30]));

        // Test SubAssign (in-place)
        v1 -= v2;
        assert_eq!(v1, expected);
    }

    #[test]
    fn test_vector_scalar_multiplication() {
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);

        // Test `Mul` (out-of-place)
        let v2 = v1 * 10;

        // Original vector should be unchanged
        assert_eq!(v1[0], 1);
        assert_eq!(v1[1], 2);
        assert_eq!(v1[2], 3);

        // New vector should have scaled values
        assert_eq!(v2[0], 10);
        assert_eq!(v2[1], 20);
        assert_eq!(v2[2], 30);

        // Test `MulAssign` (in-place)
        v1 *= 5;

        // Original vector should now be changed
        assert_eq!(v1[0], 5);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 15);
    }

    #[test]
    fn test_vector_as_ref() {
        let v = Vector::from([1, 2, 3, 4]);

        // Use a slice method that we get via AsRef
        let sum: i32 = v.as_ref().iter().sum();
        assert_eq!(sum, 10);
    }

    #[test]
    fn test_vector_as_mut() {
        let mut v = Vector::from([3, 1, 2]);

        // Use a mutable slice method that we get via AsMut
        v.as_mut().sort();
        assert_eq!(v, Vector::from([1, 2, 3]));
    }

    #[test]
    fn test_vector_neg() {
        let v1 = Vector::from([1, -2, 3]);
        let v_neg = -v1;

        let expected = Vector::from([-1, 2, -3]);
        assert_eq!(v_neg, expected);

        // Test that original is unchanged
        assert_eq!(v1, Vector::from([1, -2, 3]));
    }

    #[test]
    fn test_vector_dot_product() {
        let v1 = Vector::from([1.0, 2.0, 3.0]);
        let v2 = Vector::from([4.0, 5.0, 6.0]);

        // (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32.0
        assert_eq!(v1.dot(v2), 32.0);

        // Test with integers
        let v_int1 = Vector::from([1, 2, 3]);
        let v_int2 = Vector::from([10, 20, 30]);

        // 10 + 40 + 90 = 140
        assert_eq!(v_int1.dot(v_int2), 140);
    }

    #[test]
    fn test_vector_transpose() {
        let v = Vector::from([1.0, 2.0, 3.0]); // Our 1x3-like Vector

        // Transpose it to a 3x1 column matrix
        let col_matrix = v.transpose();

        let expected = Matrix::from([[1.0], [2.0], [3.0]]);

        assert_eq!(col_matrix, expected);
        // Check dimensions (optional, enforced by type system)
        assert_eq!(col_matrix.data.len(), 3); // 3 rows
        assert_eq!(col_matrix.data[0].data.len(), 1); // 1 column
    }

    #[test]
    fn test_matrix_indexing_syntax() {
        let mut m: Matrix<i32, 3, 4> = Matrix::default();

        // Test the `m[i][j]` write syntax
        m[1][2] = 100;

        // Test the `m[i][j]` read syntax
        assert_eq!(m[1][2], 100);

        // Test that the rest of the matrix is untouched
        assert_eq!(m[0][0], 0);
        assert_eq!(m[1][1], 0);
        assert_eq!(m[1][3], 0);
        assert_eq!(m[2][3], 0);
    }

    #[test]
    fn test_matrix_row_access() {
        let mut m: Matrix<f64, 2, 3> = Matrix::default();
        m[0][0] = 1.1;
        m[0][1] = 1.2;
        m[0][2] = 1.3;

        // Accessing `m[i]` should return the whole vector (row)
        let row0 = m[0];
        assert_eq!(row0[0], 1.1);
        assert_eq!(row0[1], 1.2);
        assert_eq!(row0[2], 1.3);
    }

    #[test]
    fn test_matrix_add_and_add_assign() {
        let mut m1 = Matrix::from([[1, 2], [3, 4]]);
        let m2 = Matrix::from([[10, 20], [30, 40]]);

        // Test Add (out-of-place)
        let m3 = m1 + m2;
        let expected = Matrix::from([[11, 22], [33, 44]]);
        assert_eq!(m3, expected);

        // Test AddAssign (in-place)
        m1 += m2;
        assert_eq!(m1, expected);
    }

    #[test]
    fn test_matrix_sub_and_sub_assign() {
        let mut m1 = Matrix::from([[11, 22], [33, 44]]);
        let m2 = Matrix::from([[1, 2], [3, 4]]);

        // Test Sub (out-of-place)
        let m3 = m1 - m2;
        let expected = Matrix::from([[10, 20], [30, 40]]);
        assert_eq!(m3, expected);

        // Test that original is unchanged
        assert_eq!(m1, Matrix::from([[11, 22], [33, 44]]));

        // Test SubAssign (in-place)
        m1 -= m2;
        assert_eq!(m1, expected);
    }

    #[test]
    fn test_matrix_mul_square() {
        let m1: Matrix<i32, 2, 2> = Matrix::from([[1, 2], [3, 4]]);
        let m2: Matrix<i32, 2, 2> = Matrix::from([[5, 6], [7, 8]]);

        let m3 = m1 * m2;
        // Expected:
        // [ (1*5 + 2*7), (1*6 + 2*8) ] = [ (5 + 14), (6 + 16) ] = [19, 22]
        // [ (3*5 + 4*7), (3*6 + 4*8) ] = [ (15 + 28), (18 + 32) ] = [43, 50]

        assert_eq!(m3[0][0], 19);
        assert_eq!(m3[0][1], 22);
        assert_eq!(m3[1][0], 43);
        assert_eq!(m3[1][1], 50);
    }

    #[test]
    fn test_matrix_mul_non_square() {
        // M=2, N=3
        let m1: Matrix<i32, 2, 3> = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        // N=3, K=2
        let m2: Matrix<i32, 3, 2> = Matrix::from([[7, 8], [9, 10], [11, 12]]);

        let m3 = m1 * m2; // Output should be M x K (2 x 2)
        // Expected:
        // [ (1*7 + 2*9 + 3*11), (1*8 + 2*10 + 3*12) ] = [ (7 + 18 + 33), (8 + 20 + 36) ] = [58, 64]
        // [ (4*7 + 5*9 + 6*11), (4*8 + 5*10 + 6*12) ] = [ (28 + 45 + 66), (32 + 50 + 72) ] = [139, 154]

        assert_eq!(m3[0][0], 58);
        assert_eq!(m3[0][1], 64);
        assert_eq!(m3[1][0], 139);
        assert_eq!(m3[1][1], 154);
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        let mut m1: Matrix<i32, 2, 2> = Matrix::from([[1, 2], [3, 4]]);

        // Test `Mul` (out-of-place)
        let m2 = m1 * 10;

        // Original matrix should be unchanged
        assert_eq!(m1[0][0], 1);
        assert_eq!(m1[0][1], 2);
        assert_eq!(m1[1][0], 3);
        assert_eq!(m1[1][1], 4);

        // New matrix should have scaled values
        assert_eq!(m2[0][0], 10);
        assert_eq!(m2[0][1], 20);
        assert_eq!(m2[1][0], 30);
        assert_eq!(m2[1][1], 40);

        // Test `MulAssign` (in-place)
        m1 *= 5;

        // Original matrix should now be changed
        assert_eq!(m1[0][0], 5);
        assert_eq!(m1[0][1], 10);
        assert_eq!(m1[1][0], 15);
        assert_eq!(m1[1][1], 20);
    }

    #[test]
    fn test_matrix_neg() {
        let m1 = Matrix::from([[1, -2], [-3, 4]]);
        let m_neg = -m1;

        let expected = Matrix::from([[-1, 2], [3, -4]]);
        assert_eq!(m_neg, expected);
    }

    #[test]
    fn test_matrix_transpose_square() {
        // A square 2x2 matrix
        let m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);

        let m_transposed = m.transpose();

        let expected = Matrix::from([[1.0, 3.0], [2.0, 4.0]]);

        assert_eq!(m_transposed, expected);
    }

    #[test]
    fn test_matrix_transpose_non_square() {
        // A non-square 2x3 matrix
        let m = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        // Transpose it to a 3x2 matrix
        let m_transposed = m.transpose();

        let expected = Matrix::from([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);

        assert_eq!(m_transposed, expected);
    }

    #[test]
    fn test_matrix_cross_product() {
        // i = [1, 0, 0]
        let v_i = Matrix::from([[1.0], [0.0], [0.0]]);
        // j = [0, 1, 0]
        let v_j = Matrix::from([[0.0], [1.0], [0.0]]);
        // k = [0, 0, 1]
        let v_k = Matrix::from([[0.0], [0.0], [1.0]]);

        // Test i x j = k
        let result_k = v_i.cross(&v_j);
        assert_eq!(result_k, v_k);

        // Test j x i = -k
        let result_neg_k = v_j.cross(&v_i);
        assert_eq!(result_neg_k, -v_k);

        // Test a x a = 0
        let result_zero = v_i.cross(&v_i);
        let expected_zero = Matrix::from([[0.0], [0.0], [0.0]]);
        assert_eq!(result_zero, expected_zero);
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        // Test Matrix * ColumnVector
        let m = Matrix::from([[1, 2, 3], [4, 5, 6]]); // 2x3 matrix

        let v = Vector::from([10, 20, 30]);
        let v_col = v.transpose(); // 3x1 column matrix

        let result = m * v_col; // 2x1 result
        let expected = Matrix::from([[140], [320]]);
        assert_eq!(result, expected);

        // Test RowVector * Matrix
        let v_row = Matrix::from(v); // 1x3 row matrix
        let m2 = Matrix::from([[1, 4], [2, 5], [3, 6]]); // 3x2 matrix

        let result2 = v_row * m2; // 1x2 result
        let expected2 = Matrix::from([[140, 320]]);
        assert_eq!(result2, expected2);
    }

    #[test]
    fn test_from_vector_to_row_matrix() {
        let v = Vector::from([1, 2, 3]);

        // This conversion creates a 1x3 row matrix
        let row_matrix = Matrix::from(v);

        let expected = Matrix::from([[1, 2, 3]]);
        assert_eq!(row_matrix, expected);
    }

    // #[test]
    // fn test_compile_time_dimension_mismatch() {
    //     // This test demonstrates the compile-time safety.
    //     // If you uncomment the `let m3 = ...` line, this test will fail to compile,
    //     // which is the *desired behavior*.
    //
    //     let m1: Matrix<i32, 2, 3> = Matrix::default(); // 2x3
    //     let m2: Matrix<i32, 4, 2> = Matrix::default(); // 4x2
    //
    //     // let m3 = m1 * m2;
    //     // ^^^
    //     // This line will fail to compile with an error like:
    //     // "mismatched types: expected `Matrix<i32, 3, _>`, found `Matrix<i32, 4, 2>`"
    //     // This is the library working as intended!
    //
    //     // We'll just assert true to make this test "pass"
    //     assert!(true, "Compile-fail test is for demonstration only");
    // }
}
