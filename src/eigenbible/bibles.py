"""Which bibles exist and where their chapter files / vector collections live.

Shared between embed_biblia.py (writes these collections) and kpca_label.py
(reads them back) - kept as plain data, not settings, since it's not
environment-specific.
"""

# key -> (source directory name under the eigenbible project root, vector
# collection name to store its vectors in)
BIBLES = {
    "ortodox": ("biblia-ortdx-capitole", "biblia_ortodoxa_subcapitole"),
    "darby": ("biblia-darby-fr-capitole", "biblia_darby_fr_subcapitole"),
}
COMBINED_COLLECTION = "biblia_all_subcapitole"

# collection name -> bible key, for collections holding a single bible's
# chapters (the combined collection instead carries a 'bible' field per row).
COLLECTION_TO_BIBLE = {collection: key for key, (_dirname, collection) in BIBLES.items()}
