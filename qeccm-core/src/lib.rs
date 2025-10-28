use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

fn basic_swap_routing() {

    // DAG
    // coupling map

    // path = self.coupling_map.shortest_undirected_path(q0, q1)

}

/// A Python module implemented in Rust.
#[pymodule]
fn qeccm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
