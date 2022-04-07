package qmilvus

import (
	"context"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

func (c *Collection) Drop(ctx context.Context) (err error) {
	var (
		_client client.Client
	)
	if _client, err = c.NewGrpcClient(ctx); err != nil {
		return err
	}
	defer _client.Close()
	return _client.DropCollection(ctx, c.collectionName)
}
