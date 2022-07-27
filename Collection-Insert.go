package qmilvus

import (
	"context"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

//set lastFlushTime to zero
var lastFlushTime map[string]time.Time = make(map[string]time.Time)

//检查源字段和目标字段的对应关系
//parameter structSlice may be new data or old data

func (c *Collection) Insert(ctx context.Context, modelSlice interface{}) (err error) {
	var (
		_client        client.Client
		_lastFlushTime time.Time = time.Time{}
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
	if _, err = _client.Insert(context.Background(), c.collectionName, c.partitionName, columes...); err != nil {
		return err
	}
	//auto refresh every 4 seconds
	if _lastFlushTime, _ = lastFlushTime[c.collectionName]; time.Now().Sub(_lastFlushTime) > time.Second {
		err = _client.Flush(context.Background(), c.collectionName, true)
		lastFlushTime[c.collectionName] = time.Now()
		if err != nil {
			return err
		}
	}
	err = _client.Flush(context.Background(), c.collectionName, true)
	return err
	//no need to Flush，milvus auto Flush every second,if Flush too frequently, it will create too many file segment
}
