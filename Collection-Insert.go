package qmilvus

import (
	"context"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

//检查源字段和目标字段的对应关系
//parameter structSlice may be new data or old data

func (c *Collection) Insert(ctx context.Context, modelSlice interface{}) (err error) {
	var (
		_client client.Client
	)
	//step1: convert []*model.Foo to []*Collection
	dataSlice := c.ModeSliceToEntitySlice(ctx, modelSlice, c.dataStruct)

	// insert into default partition
	if _client, err = c.NewGrpcClient(ctx); err != nil {
		return err
	}
	// in a main func, remember to close the client
	defer _client.Close()
	columes := c.BuildColumns(dataSlice)
	_, err = _client.Insert(context.Background(), c.collectionName, c.partitionName, columes...)
	return err
	//no need to Flush，milvus auto Flush every second,if Flush too frequently, it will create too many file segment
}
