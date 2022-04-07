package qmilvus

import (
	"context"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

//remove Milvus collection item using DeleteByPks
func (c *Collection) RemoveByKey(ctx context.Context, id int64) (err error) {
	milvuslient, errM := c.NewGrpcClient(ctx)
	if errM != nil {
		return errM
	}
	defer milvuslient.Close()
	return milvuslient.DeleteByPks(ctx, c.collectionName, c.partitionName, entity.NewColumnInt64("Id", []int64{id}))
}
