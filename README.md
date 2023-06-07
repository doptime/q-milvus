# q-milvus-driver-for-go
qmilvus provides the transparent way to use milvus.
## Feature: Auto build schema & Auto build index & Insert Search Remove easily

## step1. define you schema like this:
```
package qmilvus

import (
	"context"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	milvus "github.com/yangkequn/q-milvus-driver-for-go"
)

type FooEntity struct {
	Id         int64     `schema:"in,out" primarykey:"true"`
	Name       string    `schema:"in,out" max_length:"2048"`
	Vector     []float32 `dim:"384" schema:"in" index:"true"`
	Score      float32   `schema:"out"`
}

//Index: define your index field name and type here. required
func (v FooEntity) Index() (indexFieldName string, index entity.Index) {
	index, _ = entity.NewIndexIvfFlat(entity.IP, 256)
	return "Vector", index
}

var collection *milvus.Collection = milvus.NewCollection[FooEntity]("milvus.vm:19530",  "partitionName")
```
## step2. using collection, you can Insert Search or Remove
```
var models []*FooEntity
// insert operation
err:=collection.Insert(models)
// search operation
ids,scores,models,err:=collection.Search( query []float32)
// remove operation. type of ids : []int64
err:=collection.RemoveByKey(ids)
```