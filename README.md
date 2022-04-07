# q-milvus-driver-for-go
qmilvus provides the simplest way to use milvus.
## Feature
* auto build schema
* auto build index
* auto insert collection entity
* auto search 

> So fantastic as if milvus is transparent. Access milvus service can be so easy!

## what you should do?
1. define you schema like this:
```
package milvus

import (
	"context"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	qmilvus "github.com/yangkequn/q-milvus-driver-for-go"
)

type FooEntity struct {
	Id         int64     `schema:"in,out" primarykey:"true"`
	Name       string    `schema:""`
	Detail     string    `schema:""`
	Vector     []float32 `dim:"384" schema:"in" index:"true"`
	Score      float32   ``
}

//Index: define your index field name and type here. required
func (v FooEntity) Index() (indexFieldName string, index entity.Index) {
	index, _ = entity.NewIndexIvfFlat(entity.IP, 256)
	return "Vector", index
}

//BuildSearchVector: Calculate vector here; if precalculated, Just return it
func (v FooEntity) BuildSearchVector(ctx context.Context) (Vector []float32) {
	//text := fmt.Sprintf("Name:%s Detail:%s", v.Name, v.Detail)
	//vector, _ = Foo.CalculateVector(ctx,  text)
    //return vector
	return v.Vector
}

var FooContext *qmilvus.MilvusContext = qmilvus.MilvusContext{}.Init("milvus.vm:19530", FooEntity{}, "partitionName")
```
2. using FooContext, you can do the the left things easily:

* insert entities
```
err:=FooContext.Insert(c context.Context, bar []*Bar)
```
> here struct Bar will cast （borrow from c++ cast ） into struct FooEntity.  structs  fields with same name and type will be copied, Other Bar fields will be neglected. 

> The casted FooEntity will insert into milvus Collection according to go Tag.

> Id         int64      `schema:"in,out" primarykey:"true"`, means Id Field will insert to collection, and will returned in searchField.

* search
```
ids,scores,err:=FooContext.Search(ctx context.Context, query []float32)
```
* Remove entity
```
err:=FooContext.RemoveByKey(c context.Context, bar []*Bar)
```