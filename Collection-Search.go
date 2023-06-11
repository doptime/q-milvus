package qmilvus

import (
	"fmt"
	"reflect"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func (c *Collection[v]) SearchVector(query []float32, TopK int) (Ids []int64, Scores []float32, models []*v, err error) {
	var (
		results []client.SearchResult
		_client client.Client
	)
	if _client, err = c.NewGrpcClient(c.ctx); err != nil {
		return nil, nil, nil, err
	}
	defer _client.Close()

	//查询最相近的相似度
	vectors, vectorField := []entity.Vector{entity.FloatVector(query)}, c.IndexFieldName
	//LoadCollection is necessary
	err = _client.LoadCollection(c.ctx, c.collectionName, false)
	if err != nil {
		return nil, nil, nil, err
	}
	// Use flat search param
	searchParam, _ := entity.NewIndexFlatSearchParam()
	if results, err = _client.Search(c.ctx, c.collectionName, []string{c.partitionName}, "", c.outputFields, vectors, vectorField, entity.IP, TopK, searchParam); err != nil {
		return nil, nil, nil, err
	}

	//get Two column Id and Score, and return
	Ids, Scores = make([]int64, 0, results[0].ResultCount), results[0].Scores
	// type of model is v
	for _, field := range results[0].Fields {
		if c, ok := field.(*entity.ColumnInt64); ok && field.Name() != "Id" {
			Ids = append(Ids, c.Data()...)
		}
	}
	models, err = c.ParseSearchResult(&results[0])
	return Ids, Scores, models, nil
}
func (c *Collection[v]) SetModelFields(column entity.Column, models []*v) {
	if source, ok := column.(*entity.ColumnInt64); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.SetInt(v)
		}
	} else if source, ok := column.(*entity.ColumnVarChar); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.SetString(v)
		}
	} else if source, ok := column.(*entity.ColumnString); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.SetString(v)
		}
	} else if source, ok := column.(*entity.ColumnFloat); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.Set(reflect.ValueOf(v))
		}
	} else if source, ok := column.(*entity.ColumnDouble); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.SetFloat(v)
		}
	} else if source, ok := column.(*entity.ColumnBool); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.SetBool(v)
		}
	} else if source, ok := column.(*entity.ColumnInt8); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.Set(reflect.ValueOf(v))
		}
	} else if source, ok := column.(*entity.ColumnInt16); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.Set(reflect.ValueOf(v))
		}
	} else if source, ok := column.(*entity.ColumnInt32); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.Set(reflect.ValueOf(v))
		}
	} else if source, ok := column.(*entity.ColumnBinaryVector); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.SetBytes(v)
		}
	} else {
		panic(fmt.Errorf("unsupported type %s", column.Name()))
	}
}

func (c *Collection[v]) ParseSearchResult(result *client.SearchResult) (models []*v, err error) {
	models = make([]*v, 0, result.ResultCount)
	for i := 0; i < result.ResultCount; i++ {
		//create instance of type v
		_v := reflect.New(reflect.TypeOf((*v)(nil)).Elem()).Interface().(*v)
		models = append(models, _v)
	}
	if result.ResultCount == 0 || len(result.Fields) == 0 {
		return models, nil
	}
	// typ := v.Type()
	//extract result from search result to SearchResultColumns
	for _, field := range result.Fields {
		c.SetModelFields(field, models)
	}
	return models, nil
}
