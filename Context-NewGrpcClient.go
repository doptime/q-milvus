package qmilvus

import (
	"context"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

// NewGrpcClient : return a client with collection loaded
// data loaded to memory every 10 minutes
func (c *Collection) NewGrpcClient(ctx context.Context) (_client client.Client, err error) {
	return client.NewGrpcClient(ctx, c.milvusAdress)
}
