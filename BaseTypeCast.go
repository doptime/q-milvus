package milvus

import (
	"context"
	"errors"
	"fmt"
	"reflect"
)

//ModeSliceToCollectionSlice cast slice to collection slice
//sliceIn is: []*model.Foo{ field1, field2, field3 }
//structTo is: []*Collection{ field1, field2, field3 }
//AutoFill the meaning field of Collection
func (c *CollectionContext) ModeSliceToCollectionSlice(ctx context.Context, sliceIn interface{}, structTo interface{}) (out interface{}) {
	structSlice := SliceCast(sliceIn, structTo)

	slice, _, err := GetSliceValueType(structSlice)
	if err != nil {
		panic(err.Error())
	}

	//Meaning vector is missing, so we need to calculate it
	for i := 0; i < slice.Len(); i++ {
		_v := reflect.Indirect(slice.Index(i))
		//interface is value based, so we need to get the struct value, and then set the field
		vector := _v.MethodByName("BuildSearchVector").Call([]reflect.Value{reflect.ValueOf(ctx)})[0].Interface().([]float32)
		//Indirect is wrapped with pointer, so Set can work
		_v.FieldByName(c.IndexFieldName).Set(reflect.ValueOf(vector))
	}
	return structSlice
}

//cast one struct to another struct
//input is: *struct1{ field1, field2, field3 }
//input2 is: []*struct2{ field1, field2, field3 }
//output is: []*struct1{ field1, field2, field3 }
func SliceCast(sliceIn interface{}, structTo interface{}) (out interface{}) {
	_, typeTo, err := GetStructValueType(structTo)
	if err != nil {
		panic(err.Error())
	}
	slice, _, err := GetSliceValueType(sliceIn)
	if err != nil {
		panic(err.Error())
	}
	outSliceValue := reflect.MakeSlice(reflect.SliceOf(reflect.PointerTo(typeTo)), 0, 10)
	for i := 0; i < slice.Len(); i++ {
		//expected val is: *typ1{ field1, field2, field3 }´

		val := reflect.New(typeTo)
		err = StructCast(slice.Index(i), val)
		if err != nil {
			panic(err.Error())
		}
		outSliceValue = reflect.Append(outSliceValue, val)
	}
	return outSliceValue.Interface()
}

//  v1 is reflect.Value ,it's value is struct{ field1, field2, field3 } , t1 is v1.Type()
//  v2 is reflect.Value ,it's value is struct{ field1, field2, field3 } , t2 is v1.Type()
// set filed1, field2, field3 from v1 to v2
func StructCast(v1 reflect.Value, v2 reflect.Value) (err error) {
	if v1.Kind() == reflect.Ptr {
		v1 = v1.Elem()
	}
	if v2.Kind() == reflect.Ptr {
		v2 = v2.Elem()
	}

	t1, t2 := v1.Type(), v2.Type()
	for i := 0; i < t1.NumField(); i++ {
		field1 := t1.Field(i)
		fieldName := field1.Name
		field2, field2Ok := t2.FieldByName(fieldName)

		if !field2Ok || (field1.Type != field2.Type) {
			continue
		}
		//copy field value
		v2.FieldByName(fieldName).Set(v1.Field(i))
	}
	return nil
}

func GetStructValueType(structWithOptionalPtr interface{}) (structvalue reflect.Value, structType reflect.Type, err error) {
	if structWithOptionalPtr == nil {
		err = errors.New("ErrINilStruct")
		return
	}
	v := reflect.ValueOf(structWithOptionalPtr)
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		err = errors.New("ErrNotStruct")
		return
	}
	return v, v.Type(), nil
}

func GetSliceValueType(input interface{}) (sliceValue reflect.Value, sliceType reflect.Type, err error) {
	var v reflect.Value
	if input == nil {
		err = errors.New("ErrInvalidModelSlice")
		return
	}
	// we only accept structs
	if v = reflect.ValueOf(input); v.Kind() != reflect.Slice {
		err = fmt.Errorf("input should be a slice; got %T", v)
		return
	}
	return v, v.Type(), nil
}
