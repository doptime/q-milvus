package qmilvus

import (
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// remove Milvus collection item using DeleteByPks
func (c *Collection[v]) RemoveByKey(ids []int64) (err error) {
	milvuslient, errM := c.NewGrpcClient(c.ctx)
	if errM != nil {
		return errM
	}
	defer milvuslient.Close()
	return milvuslient.DeleteByPks(c.ctx, c.collectionName, c.partitionName, entity.NewColumnInt64("Id", ids))
}
