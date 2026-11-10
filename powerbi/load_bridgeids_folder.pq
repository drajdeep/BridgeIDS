// Power Query (M) — load every BridgeIDS CSV from a folder in one go.
// Home > Transform data > New Source > Blank Query > Advanced Editor > paste, then set DataFolder.
let
    DataFolder = "C:\BridgeIDS\powerbi\data",          // <-- change to your path
    Files      = Folder.Files(DataFolder),
    Csv        = Table.SelectRows(Files, each Text.EndsWith([Name], ".csv")),
    WithTables = Table.AddColumn(Csv, "Data", each Table.PromoteHeaders(Csv.Document([Content], [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]), [PromoteAllScalars=true])),
    Result     = Table.SelectColumns(WithTables, {"Name", "Data"})
in
    Result
// Tip: right-click each row's [Data] Table > "Add as New Query" to get one table per CSV,
// or simply use Get Data > Text/CSV for the individual files listed in POWERBI_GUIDE.md.
