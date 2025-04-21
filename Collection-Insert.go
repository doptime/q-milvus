package qmilvus

import (
	"context"
	"fmt"
	"reflect"
	"strconv"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

//检查源字段和目标字段的对应关系
//parameter structSlice may be new data or old data

func (c *Collection[v]) Insert(models ...v) (err error) {
	var (
		_client client.Client
	)
	// insert into default partition
	if _client, err = c.NewGrpcClient(c.ctx); err != nil {
		return err
	}
	// in a main func, remember to close the client
	defer _client.Close()
	columes := c.BuildColumns(models...)
	if _, err = _client.Insert(context.Background(), c.collectionName, c.partitionName, columes...); err != nil {
		return err
	}
	return err
	//no need to Flush，milvus auto Flush every second,if Flush too frequently, it will create too many file segment
}

// columes is used to insert []struct to collection
// the milvus Insert method accept collection only
func (c *Collection[v]) BuildColumns(models ...v) (result []entity.Column) {
	var (
		colume entity.Column
		err    error
		dim    int = 0
	)
	//all fields of type v to columes
	result = []entity.Column{}
	for _, s := range c.schemaIn.Fields {
		if s.DataType == entity.FieldTypeDouble {
			colume = entity.NewColumnDouble(s.Name, []float64{})
		} else if s.DataType == entity.FieldTypeFloat {
			colume = entity.NewColumnFloat(s.Name, []float32{})
		} else if s.DataType == entity.FieldTypeInt64 {
			colume = entity.NewColumnInt64(s.Name, []int64{})
		} else if s.DataType == entity.FieldTypeVarChar || s.DataType == entity.FieldTypeString {
			colume = entity.NewColumnVarChar(s.Name, []string{})
			//colume = entity.NewColumnString(s.Name, []string{})
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

		for i := 0; i < len(models); i++ {
			_v := reflect.ValueOf(models[i])
			// check if _v is a pointer, if so, get the value
			for _v.Kind() == reflect.Ptr {
				_v = _v.Elem()
			}
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

		result = append(result, colume)
	}
	return result
}
