use matrix::{Matrix, Vector};

fn main() {
    println!("--- 2D Transformation Example ---");

    // 1. Define our point (10, 20)
    // We create a row Vector and transpose it to a 3x1 column matrix.
    let p = Vector::from([10.0, 20.0, 1.0]).transpose();
    println!("Original point:\n{:#?}", p);

    // 2. Create a 90-degree counter-clockwise rotation matrix (trig: cos(90)=0, sin(90)=1)
    let rotation = Matrix::from([
        // cos, -sin, 0
        [0.0, -1.0, 0.0],
        // sin,  cos, 0
        [1.0, 0.0, 0.0],
        // 0,    0,   1
        [0.0, 0.0, 1.0],
    ]);
    println!("90-Degree rotation matrix:\n{:#?}", rotation);

    // 3. Create a translation matrix to move (x by 5, y by -10)
    let translation = Matrix::from([[1.0, 0.0, 5.0], [0.0, 1.0, -10.0], [0.0, 0.0, 1.0]]);
    println!("Translation matrix (+5, -10):\n{:#?}", translation);

    // 4. Combine transformations.
    let transform = translation * rotation;
    println!("Combined transform (translate * rotate):\n{:#?}", transform);

    // 5. Apply the combined transformation to our point
    let transformed = transform * p;
    println!("Transformed point:\n{:#?}", transformed);

    // --- Check the math ---
    // Rotate (10, 20) -> (-20, 10)
    // Translate (-20, 10) -> (-20 + 5, 10 - 10) -> (-15, 0)
    // The final point should be (-15, 0, 1)
    let expected = Vector::from([-15.0, 0.0, 1.0]).transpose();

    assert_eq!(transformed, expected);
    println!("\n✅ Transformation was correct!");
}
