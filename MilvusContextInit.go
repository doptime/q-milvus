package qmilvus

import (
	"context"
	"reflect"
)

func (c MilvusContext) Init(milvusAdress string, collectionStruct Entity, partitionName string) *MilvusContext {
	c.milvusAdress = milvusAdress
	c.partitionName = partitionName
	if len(partitionName) == 0 {
		c.partitionName = "_default"
	}

	//Auto build index, modify IndexField if you want to use other index
	c.IndexFieldName, c.Index = collectionStruct.Index()
	c.collectionName = reflect.Indirect(reflect.ValueOf(collectionStruct)).Type().Name() + "s"
	c.dataStruct = collectionStruct
	c.BuildOutputFields()
	c.BuildSchema()
	c.CreateCollection(context.Background())
	return &c
}
