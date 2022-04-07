package qmilvus

import (
	"context"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type Collection struct {
	milvusAdress   string
	partitionName  string
	collectionName string

	IndexFieldName string
	Index          entity.Index

	dataStruct   Entity
	schema       *entity.Schema
	outputFields []string
}

type CollectionInterface interface {
	Init() *Collection
	BuildColumns() []entity.Column
	NewGrpcClient() (c client.Client, err error)
	RemoveByKey(partitionName string, id int64) error
	Insert(c context.Context, modelSlice interface{}) (err error)
	Search(ctx context.Context, query []float32) (Ids []int64, Scores []float32, err error)
	Drop(ctx context.Context) (err error)
	CreateCollection(ctx context.Context) (err error)
	BuildSchema() *entity.Schema
	BuildOutputFields()
}

// BuildColumns : convert a structSlice to entity.Column, according to a schema like this:
// var FooSchema *entity.Schema = &entity.Schema{
// 	CollectionName: "CollectionFoo",
// 	Description:    "collection for insert and search theme with CollectionFoo",
// 	AutoID:         false,
// 	Fields: []*entity.Field{
// 		{Name: "Id", DataType: entity.FieldTypeInt64, PrimaryKey: true, AutoID: false},
// 		{Name: "Popularity", DataType: entity.FieldTypeString, PrimaryKey: false, AutoID: false},
// 		{Name: "Meaning", DataType: entity.FieldTypeFloatVector, TypeParams: map[string]string{"dim": "384"}},
// 	},
// }
func (c *Collection) BuildColumns(structSlice interface{}) (reslt []entity.Column) {
	var (
		colume entity.Column
		err    error
		dim    int = 0
	)

	reslt = []entity.Column{}
	for _, s := range c.schema.Fields {
		if s.DataType == entity.FieldTypeDouble {
			colume = entity.NewColumnDouble(s.Name, []float64{})
		} else if s.DataType == entity.FieldTypeFloat {
			colume = entity.NewColumnFloat(s.Name, []float32{})
		} else if s.DataType == entity.FieldTypeInt64 {
			colume = entity.NewColumnInt64(s.Name, []int64{})
		} else if s.DataType == entity.FieldTypeString {
			colume = entity.NewColumnString(s.Name, []string{})
		} else if s.DataType == entity.FieldTypeFloatVector {
			if dim, err = strconv.Atoi(s.TypeParams["dim"]); err != nil {
				panic(err)
			}
			colume = entity.NewColumnFloatVector(s.Name, dim, [][]float32{})
		} else if s.DataType == entity.FieldTypeInt32 {
			colume = entity.NewColumnInt32(s.Name, []int32{})
		} else if s.DataType == entity.FieldTypeInt16 {
			colume = entity.NewColumnInt16(s.Name, []int16{})
		} else if s.DataType == entity.FieldTypeInt8 {
			colume = entity.NewColumnInt8(s.Name, []int8{})
		} else if s.DataType == entity.FieldTypeBool {
			colume = entity.NewColumnBool(s.Name, []bool{})
		} else if s.DataType == entity.FieldTypeBinaryVector {
			if dim, err = strconv.Atoi(s.TypeParams["dim"]); err != nil {
				panic(err)
			}
			colume = entity.NewColumnBinaryVector(s.Name, dim, [][]byte{})
		} else {
			panic(fmt.Sprintf("unsupported data type: %v", s.DataType))
		}

		reslt = append(reslt, colume)

		sliceValue := reflect.ValueOf(structSlice)
		for i := 0; i < sliceValue.Len(); i++ {
			_v := reflect.Indirect(sliceValue.Index(i))
			_field := _v.FieldByName(s.Name)
			// check demension match, if not, skip Insert
			if _field.Type().Kind() == reflect.Slice {
				vectorLen := _field.Len()
				if vectorLen != dim {
					println("Error: milvus insert dim not match")
				}
			}
			colume.AppendValue(_field.Interface())

		}
	}
	return reslt
}

func (c *Collection) BuildOutputFields() {
	structvalue, structType, err := GetStructValueType(c.dataStruct)
	if err != nil {
		panic(err)
	}

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

func (c *Collection) BuildSchema() {
	structvalue, structType, err := GetStructValueType(c.dataStruct)
	if err != nil {
		panic(err)
	}

	c.schema = &entity.Schema{
		CollectionName: structType.Name(),
		Description:    "collection for insert and search with " + structType.Name(),
		AutoID:         false,
		Fields:         []*entity.Field{},
	}

	primarykey := 0
	for i := 0; i < structvalue.NumField(); i++ {
		// gets us a StructField
		vi := structvalue.Field(i).Interface()
		tpi := structType.Field(i)
		if isSchema := tpi.Tag.Get("schema"); !strings.Contains(isSchema, "in") {
			continue
		}
		TypeParams := map[string]string{}

		var columeType entity.FieldType
		_primarykey := false
		if _, ok := vi.(int64); ok {
			columeType = entity.FieldTypeInt64
			if tagv := tpi.Tag.Get("primarykey"); tagv != "" {
				_primarykey = true
				primarykey += 1
			}
		} else if _, ok := vi.(string); ok {
			columeType = entity.FieldTypeString
		} else if _, ok := vi.(float32); ok {
			columeType = entity.FieldTypeFloat
		} else if _, ok := vi.(float64); ok {
			columeType = entity.FieldTypeDouble
		} else if _, ok := vi.([]float32); ok {
			columeType = entity.FieldTypeFloatVector
			if TypeParams["dim"] = tpi.Tag.Get("dim"); TypeParams["dim"] == "" {
				panic(fmt.Errorf("%s dim is not set", tpi.Name))
			}
		} else if _, ok := vi.(bool); ok {
			columeType = entity.FieldTypeBool
		} else if _, ok := vi.(int8); ok {
			columeType = entity.FieldTypeInt8
		} else if _, ok := vi.(int16); ok {
			columeType = entity.FieldTypeInt16
		} else if _, ok := vi.(int32); ok {
			columeType = entity.FieldTypeInt32
		} else if _, ok := vi.(byte); ok {
			columeType = entity.FieldTypeBinaryVector
			if TypeParams["dim"] = tpi.Tag.Get("dim"); TypeParams["dim"] == "" {
				panic(fmt.Errorf("%s dim is not set", tpi.Name))
			}
		} else {
			panic(fmt.Errorf("unsupported type %s", structType.Field(i).Type.String()))
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
