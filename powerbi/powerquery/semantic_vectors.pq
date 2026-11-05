let
    Source = Csv.Document(File.Contents(ExportFolder & "/semantic_vectors.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Nulls = Table.ReplaceValue(Headers, "", null, Replacer.ReplaceValue, Table.ColumnNames(Headers)),
    Typed = Table.TransformColumnTypes(Nulls, {{"document_id", Int64.Type}, {"title", type text}, {"domain", type text}, {"collection", type text}, {"era", type text}, {"x", type number}, {"y", type number}}, "en-US"),
    Clean = Table.ReplaceErrorValues(Typed, List.Transform(Table.ColumnNames(Typed), each {_, null}))
in
    Clean
