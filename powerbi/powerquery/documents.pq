let
    Source = Csv.Document(File.Contents(ExportFolder & "/documents.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Nulls = Table.ReplaceValue(Headers, "", null, Replacer.ReplaceValue, Table.ColumnNames(Headers)),
    Typed = Table.TransformColumnTypes(Nulls, {{"document_id", Int64.Type}, {"title", type text}, {"doc_type", type text}, {"date_written", type text}, {"digitized_at", type datetime}, {"physical_condition", type text}, {"collection_name", type text}, {"archive_name", type text}, {"era", type text}, {"author_names", type text}, {"ocr_status", type text}, {"collection_id", Int64.Type}, {"archive_id", Int64.Type}}, "en-US"),
    Clean = Table.ReplaceErrorValues(Typed, List.Transform(Table.ColumnNames(Typed), each {_, null}))
in
    Clean
