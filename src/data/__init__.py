
from .split_builders import (
    attach_slide_ids,
    make_split_indices,
)

from .split_analysis import (
    build_split_assignment_table,
    count_classes_per_slide,
    build_split_class_summary,
    summarize_split_acceptability,
)

from .split_io import (
    save_split_artifacts,
    load_split_indices,
    validate_precomputed_split,
)

# Así train_classifier.py y tune_classifier.py siguen importando limpio desde src.data.