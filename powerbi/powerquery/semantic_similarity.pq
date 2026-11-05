let
    Source = Csv.Document(File.Contents(ExportFolder & "/semantic_similarity.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Nulls = Table.ReplaceValue(Headers, "", null, Replacer.ReplaceValue, Table.ColumnNames(Headers)),
    Typed = Table.TransformColumnTypes(Nulls, {{"document_id_a", Int64.Type}, {"document_id_b", Int64.Type}, {"title_a", type text}, {"title_b", type text}, {"cosine_similarity", type number}}, "en-US"),
    Clean = Table.ReplaceErrorValues(Typed, List.Transform(Table.ColumnNames(Typed), each {_, null}))
in
    Clean
