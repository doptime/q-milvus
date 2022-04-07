package qmilvus

import (
	"context"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

//Create : try to create a collection, if it already exists, do nothing
//if you want to remove the collection if the Schema is changed
// just rename the collection name, another Collection will be created, without remove the old one
func (c *Collection) Create(ctx context.Context) (err error) {
	var (
		_client client.Client
	)
	if _client, err = c.NewGrpcClient(ctx); err != nil {
		return err
	}
	defer _client.Close()

	if err = _client.CreateCollection(ctx, c.schema, 1); err != nil {
		//if err string do not contain "already exists",return err
		if !strings.Contains(err.Error(), "already exist") {
			return err
		}
	}
	//create partition
	if err = _client.CreatePartition(ctx, c.collectionName, c.partitionName); err != nil {
		//if err string do not contain "already exists",return err
		if !strings.Contains(err.Error(), "already exists") {
			return err
		}
	}
	//Auto BuildIndex
	if len(c.IndexFieldName) > 0 {
		indexState, indexErr := _client.GetIndexState(ctx, c.collectionName, c.IndexFieldName)
		if indexErr != nil {
			return indexErr
		}
		//no index exists, create index
		if indexState == 0 {
			if err = _client.CreateIndex(ctx, c.collectionName, c.IndexFieldName, c.Index, false); err != nil {
				return err
			}
		}
	}
	return err
}
