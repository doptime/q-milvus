# q-milvus-driver-for-go
Quick milvus driver for go, provides the simplest way to use milvus.
## Feature
* auto build schema
* auto build index
* auto insert collection entity
* auto search 

> so fantastic as if milvus is transparent.Helping quickly access milvus service!

# what you should do?
1. define you schema like this:
```
type FooCollection struct {
	Id         int64     `schema:"in,out" primarykey:"true"`
	Name       string    `schema:""`
	Detail     string    `schema:""`
	Vector     []float32 `dim:"384" schema:"in" index:"true"`
	Score      float32   ``
}

//Index: return the name of index, and the index type, used to build index automatically
//index name is also used by search
func (v FooCollection) Index() (indexFieldName string, index entity.Index) {
	index, _ = entity.NewIndexIvfFlat(entity.IP, 256)
	return "Vector", index
}

//BuildSearchVector: return the vector to be Inserted
//If your Vector is precalculated, Just return it
func (v FooCollection) BuildSearchVector(ctx context.Context) (Vector []float32) {
	text := fmt.Sprintf("Name:%s Detail:%s", v.Name, v.Detail)
	Vector, _ = Foo.CalculateVector(ctx,  text)
	return Vector
}

var FooContext *CollectionContext = CollectionContext{}.Init("milvus.vm:19530", FooCollection{}, "partitionName")
```
2. using FooContext, you can do the the left things easily:

* insert entities
```
err:=FooContext.Insert(c context.Context, bar []*Bar)
```
> here struct Bar and FooCollection should  shares some fields with same Name and Type. these fields will cast from Bar to FooCollection. 

* search
```
ids,scores,err:=FooContext.Search(ctx context.Context, query []float32)
```
* Remove entity
```
err:=FooContext.RemoveByKey(c context.Context, bar []*Bar)
```