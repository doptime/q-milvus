package qmilvus

import (
	"context"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

//if Entity wants to be rebuild, just rename the collection, so the collection name changed accordingly
type Entity interface {
	BuildSearchVector(ctx context.Context) (Vector []float32)
	// Index : return the index field name and index type
	// Only one index field is allowed, because milvus search engine only support one index field
	Index() (string, entity.Index)
}
