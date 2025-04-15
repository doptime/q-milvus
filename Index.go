package qmilvus

type IndexName string

const (
	IndexFlat       IndexName = "FLAT"
	IndexIvfFlat    IndexName = "IVFFLAT"
	IndexIvfSQ8     IndexName = "IVFSQ8"
	indexIvfPQ      IndexName = "IVFPQ"
	IndexHNSW       IndexName = "HNSW"
	IndexANNOY      IndexName = "ANNOY"
	IndexBinIvfFlat IndexName = "BIN_IVFFLAT"
	IndexBinFlat    IndexName = "BIN_FLAT"
	IndexAuto       IndexName = "AUTOINDEX"
)
