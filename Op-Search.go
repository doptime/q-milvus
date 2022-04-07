package qmilvus

import (
	"context"
	"fmt"
	"reflect"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func (c *Collection) Search(ctx context.Context, query []float32) (Ids []int64, Scores []float32, err error) {
	var (
		sr      []client.SearchResult
		_client client.Client
	)
	if _client, err = c.NewMilvusClient(ctx); err != nil {
		return nil, nil, err
	}
	defer _client.Close()

	//查询最相近的相似度
	vectors, vectorField := []entity.Vector{entity.FloatVector(query)}, c.IndexFieldName
	//LoadCollection is necessary
	err = _client.LoadCollection(ctx, c.collectionName, false)
	if err != nil {
		return nil, nil, err
	}
	// Use flat search param
	sp, _ := entity.NewIndexFlatSearchParam(100)
	if sr, err = _client.Search(ctx, c.collectionName, []string{c.partitionName}, "", c.outputFields, vectors, vectorField, entity.IP, 10, sp); err != nil {
		return nil, nil, err
	}

	//get Two column Id and Score, and return
	Ids, Scores = make([]int64, 0, len(sr)), make([]float32, 0, len(sr))
	for _, result := range sr {
		for _, field := range result.Fields {
			if c, ok := field.(*entity.ColumnInt64); ok && field.Name() == "Id" {
				Ids = append(Ids, c.Data()...)
			}
		}
		Scores = append(Scores, result.Scores...)
	}
	return Ids, Scores, nil
}

func (c *Collection) ParseSearchResult(sr *[]client.SearchResult) (result interface{}, err error) {
	v := reflect.Indirect(reflect.ValueOf(c.dataStruct))
	// we only accept structs
	if v.Kind() != reflect.Struct {
		return nil, fmt.Errorf("only accepts structs; got %T", v)
	}
	// typ := v.Type()
	//extract result from search result to SearchResultColumns
	for _, result := range *sr {
		for _, field := range result.Fields {
			var des reflect.Value = v.FieldByName(field.Name())
			//continue if field is not exist in data
			if des.Kind() == reflect.Invalid {
				continue
			}

			if c, ok := field.(*entity.ColumnInt64); ok {
				if d, okd := des.Interface().([]int64); okd {
					d = append(d, c.Data()...)
				}
			} else if c, ok := field.(*entity.ColumnString); ok {
				if d, okd := des.Interface().([]string); okd {
					d = append(d, c.Data()...)
				}
			} else if c, ok := field.(*entity.ColumnFloat); ok {
				if d, okd := des.Interface().([]float32); okd {
					d = append(d, c.Data()...)
				}
			} else if c, ok := field.(*entity.ColumnDouble); ok {
				if d, okd := des.Interface().([]float64); okd {
					d = append(d, c.Data()...)
				}
			} else if c, ok := field.(*entity.ColumnBool); ok {
				if d, okd := des.Interface().([]bool); okd {
					d = append(d, c.Data()...)
				}
			} else if c, ok := field.(*entity.ColumnInt8); ok {
				if d, okd := des.Interface().([]int8); okd {
					d = append(d, c.Data()...)
				}
			} else if c, ok := field.(*entity.ColumnInt16); ok {
				if d, okd := des.Interface().([]int16); okd {
					d = append(d, c.Data()...)
				}
			} else if c, ok := field.(*entity.ColumnInt32); ok {
				if d, okd := des.Interface().([]int32); okd {
					d = append(d, c.Data()...)
				}
			} else if c, ok := field.(*entity.ColumnBinaryVector); ok {
				if d, okd := des.Interface().([][]byte); okd {
					d = append(d, c.Data()...)
				}
			} else {
				panic(fmt.Errorf("unsupported type %s", field.Name()))
			}
		}

		var des reflect.Value = v.FieldByName("Score")
		//append scores
		if d, okd := des.Interface().([]float32); okd {
			d = append(d, result.Scores...)
		}
	}
	return nil, nil
}
