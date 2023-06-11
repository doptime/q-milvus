package qmilvus

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// type v should contains fields Id Vector and Score
//
//	type FooEntity struct {
//		Id         int64     `schema:"in,out" primarykey:"true"`
//		Name       string    `schema:""`
//		Detail     string    `schema:""`
//		Vector     []float32 `dim:"384" schema:"in" index:"true"`
//		Score      float32   ``
//	}
type milvusEntity interface {
	Index() (indexFieldName string, index entity.Index)
}
type Collection[v any] struct {
	ctx            context.Context
	milvusAdress   string
	partitionName  string
	collectionName string

	IndexFieldName string
	Index          entity.Index

	schema       *entity.Schema
	outputFields []string
}

func (c *Collection[v]) WithContext(ctx context.Context) (ret *Collection[v]) {
	c.ctx = ctx
	return c
}

func (c *Collection[v]) WithIndex() (ret *Collection[v]) {
	return c
}

// index : i.g. entity.NewIndexIvfFlat(entity.IP, 768)
func NewCollection[v milvusEntity](milvusAdress string, partitionName string, CreateCollectionInServer bool) (collection *Collection[v], err error) {
	c := &Collection[v]{}
	c.milvusAdress = milvusAdress
	c.partitionName = partitionName
	c.ctx = context.Background()

	//create instance of type v
	_v := reflect.New(reflect.TypeOf((*v)(nil)).Elem()).Interface().(milvusEntity)
	c.IndexFieldName, c.Index = _v.Index()

	//take name of type v as collection name
	_type := reflect.TypeOf((*v)(nil))
	for _type.Kind() == reflect.Ptr || _type.Kind() == reflect.Slice {
		_type = _type.Elem()
	}
	c.collectionName = _type.Name() + "s"

	if len(partitionName) == 0 {
		c.partitionName = "_default"
	}
	c.BuildOutputFields()
	c.BuildInSchema()
	if CreateCollectionInServer {
		if err = c.Create(); err != nil && strings.Contains(err.Error(), "already exists") {
			err = nil
		}
	}
	return c, err
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
		if isSchema := tpi.Tag.Get("schema"); !strings.Contains(isSchema, "out") {
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

	primarykey := 0
	for i := 0; i < _type.NumField(); i++ {
		// gets us a StructField
		tpi := _type.Field(i)
		if isSchema := tpi.Tag.Get("schema"); !strings.Contains(isSchema, "in") {
			continue
		}
		TypeParams := map[string]string{}

		var columeType entity.FieldType
		_primarykey := false
		_fieldType := tpi.Type.String()
		if _fieldType == "int64" {
			columeType = entity.FieldTypeInt64
			if tagvalue = tpi.Tag.Get("primarykey"); tagvalue != "" {
				_primarykey = true
				primarykey += 1
			}
		} else if _fieldType == "string" {
			columeType = entity.FieldTypeVarChar
			if tagvalue := tpi.Tag.Get("primarykey"); tagvalue != "" {
				_primarykey = true
				primarykey += 1
			}
			if tagvalue = tpi.Tag.Get(entity.TypeParamMaxLength); tagvalue == "" {
				panic(fmt.Errorf("%s %s is not set", tpi.Name, entity.TypeParamMaxLength))
			}
			TypeParams[entity.TypeParamMaxLength] = tagvalue
		} else if _fieldType == "float32" {
			columeType = entity.FieldTypeFloat
		} else if _fieldType == "float64" {
			columeType = entity.FieldTypeDouble
		} else if _fieldType == "[]float32" {
			columeType = entity.FieldTypeFloatVector
			if tagvalue = tpi.Tag.Get("dim"); tagvalue == "" {
				panic(fmt.Errorf("%s dim is not set", tpi.Name))
			}
			TypeParams[entity.TypeParamDim] = tagvalue
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
			if TypeParams["dim"] = tpi.Tag.Get("dim"); TypeParams["dim"] == "" {
				panic(fmt.Errorf("%s dim is not set", tpi.Name))
			}
		} else {
			panic(fmt.Errorf("unsupported type %s", _fieldType))
		}

		c.schema.Fields = append(c.schema.Fields, &entity.Field{
			Name:       tpi.Name,
			DataType:   columeType,
			PrimaryKey: _primarykey,
			AutoID:     false,
			TypeParams: TypeParams,
		})

	}
	if primarykey != 1 {
		panic(fmt.Errorf("primarykey not unique"))
	}
}
