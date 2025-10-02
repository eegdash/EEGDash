# This file will contain the custom sphinx-gallery logic.
from sphinx_gallery import gen_rst
from sphinx_time_estimation import get_reading_time

# Save the original function before overriding it
_original_save_rst_example = gen_rst.save_rst_example


def save_rst_example_wrapper(
    example_rst,
    example_file,
    time_elapsed,
    memory_used,
    gallery_conf,
    **kwargs,
):
    """
    Wraps the original save_rst_example to add reading time.
    """
    # Calculate reading time
    reading_time = get_reading_time(example_rst)

    # Add reading time to the rst output
    header = f"**Estimated reading time:** {reading_time} minutes\n\n"
    example_rst = header + example_rst

    # Call the original function
    _original_save_rst_example(
        example_rst,
        example_file,
        time_elapsed,
        memory_used,
        gallery_conf,
        **kwargs,
    )

# Override the original function with our custom wrapper
gen_rst.save_rst_example = save_rst_example_wrapper