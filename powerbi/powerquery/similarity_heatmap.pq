let
    Source = semantic_similarity,
    Mirrored = Table.RenameColumns(Source, {{"document_id_a", "temp_id"}, {"title_a", "temp_title"}, {"document_id_b", "document_id_a"}, {"title_b", "title_a"}}),
    Renamed = Table.RenameColumns(Mirrored, {{"temp_id", "document_id_b"}, {"temp_title", "title_b"}}),
    Result = Table.Combine({Source, Renamed})
in
    Result
