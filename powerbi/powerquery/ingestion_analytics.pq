let
    Source = Csv.Document(File.Contents(ExportFolder & "/ingestion_analytics.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Nulls = Table.ReplaceValue(Headers, "", null, Replacer.ReplaceValue, Table.ColumnNames(Headers)),
    Typed = Table.TransformColumnTypes(Nulls, {{"job_id", type text}, {"scan_id", type text}, {"device_name", type text}, {"status", type text}, {"created_at", type datetime}, {"completed_at", type datetime}, {"processing_duration_seconds", type number}}, "en-US"),
    Clean = Table.ReplaceErrorValues(Typed, List.Transform(Table.ColumnNames(Typed), each {_, null}))
in
    Clean
