package qmilvus

import (
	"fmt"
	"reflect"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// remove Milvus collection item using DeleteByPks
func (c *Collection[v]) RemoveByKeysI64(ids ...int64) (err error) {
	milvuslient, errM := c.NewGrpcClient(c.ctx)
	if errM != nil {
		return errM
	}
	defer milvuslient.Close()
	return milvuslient.DeleteByPks(c.ctx, c.collectionName, c.partitionName, entity.NewColumnInt64("Id", ids))
}

// remove Milvus collection item using DeleteByPks
func (c *Collection[v]) RemoveByKeysString(ids ...string) (err error) {
	milvuslient, errM := c.NewGrpcClient(c.ctx)
	if errM != nil {
		return errM
	}
	defer milvuslient.Close()
	return milvuslient.DeleteByPks(c.ctx, c.collectionName, c.partitionName, entity.NewColumnString("Id", ids))
}
func (c *Collection[v]) Remove(values ...v) (err error) {
	milvuslient, errM := c.NewGrpcClient(c.ctx)
	if errM != nil {
		return errM
	}
	defer milvuslient.Close()

	//take name of type v as collection name
	_type := reflect.TypeOf((*v)(nil))
	for _type.Kind() == reflect.Ptr || _type.Kind() == reflect.Slice {
		_type = _type.Elem()
	}
	// get field name of type v
	pkField, ok := _type.FieldByName(c.pkFieldName)
	if !ok || pkField.Type.Kind() != reflect.Int64 && pkField.Type.Kind() != reflect.String {
		return fmt.Errorf("pk field %s not found in type %s", c.pkFieldName, _type.Name())
	}
	// pk type should be int64 or string
	if pkField.Type.Kind() == reflect.Int64 {
		ids := make([]int64, 0)
		for _, v := range values {
			vv := reflect.ValueOf(v)
			for vv.Kind() == reflect.Ptr || vv.Kind() == reflect.Slice {
				vv = vv.Elem()
			}
			// get field value of type v
			fieldValue := vv.FieldByName(c.pkFieldName)
			ids = append(ids, fieldValue.Int())
		}
		return milvuslient.DeleteByPks(c.ctx, c.collectionName, c.partitionName, entity.NewColumnInt64("Id", ids))
	} else if pkField.Type.Kind() == reflect.String {
		ids := make([]string, 0)
		for _, v := range values {
			vv := reflect.ValueOf(v)
			for vv.Kind() == reflect.Ptr || vv.Kind() == reflect.Slice {
				vv = vv.Elem()
			}
			// get field value of type v
			fieldValue := reflect.ValueOf(vv).FieldByName(c.pkFieldName)
			ids = append(ids, fieldValue.String())
		}
		return milvuslient.DeleteByPks(c.ctx, c.collectionName, c.partitionName, entity.NewColumnString("Id", ids))
	} else {
		return fmt.Errorf("pk field %s type %s not supported", c.pkFieldName, pkField.Type.Kind())
	}
}
