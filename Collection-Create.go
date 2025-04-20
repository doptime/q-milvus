package qmilvus

import (
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/rs/zerolog/log"
)

// CreateCollection : try to create a collection, if it already exists, do nothing
// CreateCollection Only needs to be called once ever
// if you want to remove the collection ,just rename the collection name, and remove manually in attu
func (c *Collection[v]) CreateCollection() (ret *Collection[v]) {
	var (
		_client    client.Client
		indexState entity.IndexState
		err        error
	)
	if _client, err = c.NewGrpcClient(c.ctx); err != nil {
		log.Panic().Str("cannot connect milvus", c.milvusAddress)
	}
	defer _client.Close()

	if err = _client.CreateCollection(c.ctx, c.schemaIn, 1); err != nil {
		//if err string do not contain "already exists",return err
		if !strings.Contains(err.Error(), "already exist") {
			log.Panic().Err(err)
		}
	}
	//create partition
	if err = _client.CreatePartition(c.ctx, c.collectionName, c.partitionName); err != nil {
		//if err string do not contain "already exists",return err
		if !strings.Contains(err.Error(), "already exists") {
			log.Panic().Err(err)
		}
	}
	//Auto BuildIndex
	if c.IndexFieldName != "" && c.Index != nil {
		if indexState, err = _client.GetIndexState(c.ctx, c.collectionName, c.IndexFieldName); err != nil {
			log.Panic().Err(err)
		}
		//no index exists, create index
		if indexState == 0 {
			if err = _client.CreateIndex(c.ctx, c.collectionName, c.IndexFieldName, c.Index, false); err != nil {
				log.Panic().Err(err)
			}
		}
	}
	return c
}
