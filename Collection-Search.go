package qmilvus

import (
	"fmt"
	"reflect"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func (c *Collection[v]) Search(query []float32) (Ids []int64, Scores []float32, models []v, err error) {
	var (
		sr      []client.SearchResult
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
	if sr, err = _client.Search(c.ctx, c.collectionName, []string{c.partitionName}, "", c.outputFields, vectors, vectorField, entity.IP, 10, searchParam); err != nil {
		return nil, nil, nil, err
	}

	//get Two column Id and Score, and return
	Ids, Scores = make([]int64, 0, len(sr)), make([]float32, 0, len(sr))
	for _, result := range sr {
		// type of model is v
		for _, field := range result.Fields {
			if c, ok := field.(*entity.ColumnInt64); ok && field.Name() == "Id" {
				Ids = append(Ids, c.Data()...)
			}
		}
		Scores = append(Scores, result.Scores...)
	}
	models, err = c.ParseSearchResult(sr)
	return Ids, Scores, models, nil
}
func (c *Collection[v]) SetModelFields(column entity.Column, models []v) {
	if source, ok := column.(*entity.ColumnInt64); ok {
		for i, v := range source.Data() {
			model := reflect.ValueOf(models[i]).Elem()
			var des reflect.Value = model.FieldByName(source.Name())
			des.SetInt(v)
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

func (c *Collection[v]) ParseSearchResult(sr []client.SearchResult) (models []v, err error) {
	models = make([]v, 0, len(sr))
	// typ := v.Type()
	//extract result from search result to SearchResultColumns
	for _, result := range sr {
		for _, field := range result.Fields {
			if field.Len() == 0 {
				continue
			}
			modelv := models[0]
			model := reflect.ValueOf(modelv).Elem()
			var des reflect.Value = model.FieldByName(field.Name())
			//continue if field is not exist in data
			if des.Kind() == reflect.Invalid {
				continue
			}
			//continue if type of des and type of field is not equal
			if des.Kind().String() != field.Type().String() {
				continue
			}
			c.SetModelFields(field, models)
		}
	}
	return models, nil
}
