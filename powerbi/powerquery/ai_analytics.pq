let
    Source = Csv.Document(File.Contents(ExportFolder & "/ai_analytics.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Nulls = Table.ReplaceValue(Headers, "", null, Replacer.ReplaceValue, Table.ColumnNames(Headers)),
    Typed = Table.TransformColumnTypes(Nulls, {{"document_id", Int64.Type}, {"document_title", type text}, {"ocr_engine", type text}, {"ocr_confidence", type number}, {"predicted_condition", type text}, {"prediction_confidence", type number}, {"model_name", type text}, {"correction_status", type text}}, "en-US"),
    Clean = Table.ReplaceErrorValues(Typed, List.Transform(Table.ColumnNames(Typed), each {_, null}))
in
    Clean
