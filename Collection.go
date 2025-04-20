package qmilvus

import (
	"context"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// type v should contains fields Id Vector and Score
//
//	type FooEntity struct {
//		Id         int64     `milvus:"in,out,PK,dim"`
//		Name       string    `milvus:""`
//		Detail     string    `milvus:""`
//		Vector     []float32 `milvus:"dim=384,index"`
//		Ogg    	   string    `milvus:"in,out,max_length=65535"`
//		Score      float32   ``
//	}
type Collection[v any] struct {
	ctx            context.Context
	milvusAddress  string
	partitionName  string
	collectionName string

	IndexFieldName string
	Index          entity.Index

	pkFieldName  string // 主键字段名 (通常由 Schema 定义)
	schema       *entity.Schema
	outputFields []string

	client     client.Client
	clientOnce sync.Once
	clientErr  error
}

func (c *Collection[v]) WithContext(ctx context.Context) (ret *Collection[v]) {
	c.ctx = ctx
	return c
}

// index : i.g. entity.NewIndexIvfFlat(entity.IP, 768)
func NewCollection[v any](milvusAdress string) (collection *Collection[v]) {
	c := &Collection[v]{}

	if !strings.Contains(milvusAdress, ":") {
		milvusAdress = milvusAdress + ":19530"
	}
	c.milvusAddress = milvusAdress
	c.partitionName = "_default"
	c.ctx = context.Background()

	//take name of type v as collection name
	_type := reflect.TypeOf((*v)(nil))
	for _type.Kind() == reflect.Ptr || _type.Kind() == reflect.Slice {
		_type = _type.Elem()
	}
	c.collectionName = _type.Name() + "s"

	c.BuildOutputFields()
	c.BuildInSchema()
	return c
}
func (collection *Collection[v]) WithPartitionName(partitionName string) (ret *Collection[v]) {
	collection.partitionName = partitionName
	return collection
}
func (collection *Collection[v]) WithCollectionName(collectionName string) (ret *Collection[v]) {
	collection.collectionName = collectionName
	return collection
}
func (collection *Collection[v]) WithCreateIndex(index entity.Index) (ret *Collection[v]) {
	collection.Index = index
	return collection
}

func (c *Collection[v]) BuildOutputFields() {
	var (
		structvalue reflect.Value
		structType  reflect.Type
	)
	structType = reflect.TypeOf((*v)(nil))
	for structType.Kind() == reflect.Ptr || structType.Kind() == reflect.Slice {
		structType = structType.Elem()
	}
	structvalue = reflect.New(structType).Elem()

	c.outputFields = []string{}
	for i := 0; i < structvalue.NumField(); i++ {
		// gets us a StructField
		tpi := structType.Field(i)
		if isSchema := tpi.Tag.Get("milvus"); !strings.Contains(isSchema, "out") {
			continue
		}
		c.outputFields = append(c.outputFields, tpi.Name)
	}
}

func (c *Collection[v]) BuildInSchema() {
	var (
		tagvalue string
	)
	_type := reflect.TypeOf((*v)(nil))
	for _type.Kind() == reflect.Ptr || _type.Kind() == reflect.Slice {
		_type = _type.Elem()
	}

	c.schema = &entity.Schema{
		CollectionName: c.collectionName,
		Description:    "collection of " + _type.Name() + "s",
		AutoID:         false,
		Fields:         []*entity.Field{},
	}

	for i := 0; i < _type.NumField(); i++ {
		// gets us a StructField
		tpi := _type.Field(i)
		tagMilvus := strings.ToLower(tpi.Tag.Get("milvus"))
		if !strings.Contains(tagMilvus, "in") {
			continue
		}
		TypeParams := map[string]string{}
		_fieldType := tpi.Type.String()
		_primarykey := strings.Contains(tagMilvus, "PK") && (_fieldType == "int64" || _fieldType == "string")
		if _primarykey {
			if c.pkFieldName != "" {
				panic(fmt.Errorf("primarykey should be unique, only one field can be set as primary key"))
			}
			c.pkFieldName = tpi.Name
		}

		var columeType entity.FieldType
		if _fieldType == "int64" {
			columeType = entity.FieldTypeInt64
		} else if _fieldType == "string" {
			columeType = entity.FieldTypeVarChar
			if TypeParams[entity.TypeParamMaxLength] = tpi.Tag.Get(entity.TypeParamMaxLength); TypeParams[entity.TypeParamMaxLength] == "" {
				TypeParams[entity.TypeParamMaxLength] = "65535"
			}
		} else if _fieldType == "float32" {
			columeType = entity.FieldTypeFloat
		} else if _fieldType == "float64" {
			columeType = entity.FieldTypeDouble
		} else if _fieldType == "[]float32" {
			columeType = entity.FieldTypeFloatVector
			//set `dim`  `max_capacity`
			if _, val, ok := strings.Cut(tagMilvus, entity.TypeParamDim+"="); ok {
				TypeParams[entity.TypeParamDim] = strings.TrimSuffix(val, ", ")
				if _, err := strconv.Atoi(TypeParams[entity.TypeParamDim]); err != nil {
					panic(fmt.Errorf("%s %s is not set", tpi.Name, TypeParams[entity.TypeParamDim]))
				}
			}

		} else if _fieldType == "bool" {
			columeType = entity.FieldTypeBool
		} else if _fieldType == "int8" {
			columeType = entity.FieldTypeInt8
		} else if _fieldType == "int16" {
			columeType = entity.FieldTypeInt16
		} else if _fieldType == "int32" {
			columeType = entity.FieldTypeInt32
		} else if _fieldType == "byte" {
			columeType = entity.FieldTypeBinaryVector
			if strings.Contains(tagvalue, "ind") {
				c.IndexFieldName = tpi.Name
			}
			//set `dim`  `max_capacity`
			if _, val, ok := strings.Cut(tagMilvus, entity.TypeParamDim+"="); ok {
				TypeParams[entity.TypeParamDim] = strings.TrimSuffix(val, ", ")
				if _, err := strconv.Atoi(TypeParams[entity.TypeParamDim]); err != nil {
					panic(fmt.Errorf("%s %s is not set", tpi.Name, TypeParams[entity.TypeParamDim]))
				}
			}
		} else {
			panic(fmt.Errorf("PrimaryKey should be unique, with type int64 or string, unsupported type %s", _fieldType))
		}

		c.schema.Fields = append(c.schema.Fields, &entity.Field{
			Name:       tpi.Name,
			DataType:   columeType,
			PrimaryKey: _primarykey,
			AutoID:     false,
			TypeParams: TypeParams,
		})

	}

}
