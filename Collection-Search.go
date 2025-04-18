package qmilvus

import (
	"fmt"
	"reflect"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func (c *Collection[v]) SearchVector(query []float32, TopK int) (Ids []int64, Scores []float32, models []v, err error) {
	var (
		results []client.SearchResult
	)

	client, err := c.getClient()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("get client failed: %w", err)
	}

	//查询最相近的相似度
	vectors, vectorField := []entity.Vector{entity.FloatVector(query)}, c.IndexFieldName
	//LoadCollection is necessary
	err = client.LoadCollection(c.ctx, c.collectionName, false)
	if err != nil {
		return nil, nil, nil, err
	}
	// Use flat search param
	searchParam, _ := entity.NewIndexFlatSearchParam()
	if results, err = client.Search(c.ctx, c.collectionName, []string{c.partitionName}, "", c.outputFields, vectors, vectorField, entity.IP, TopK, searchParam); err != nil {
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
	models, err = c.ParseSearchResult(&results[0], Ids)
	return Ids, Scores, models, err
}
func (c *Collection[v]) ParseSearchResult(result *client.SearchResult, ids []int64) (models []v, err error) {
	resultCount := result.ResultCount
	if resultCount == 0 {
		return []v{}, nil
	}

	models = make([]v, resultCount)

	vType := reflect.TypeOf((*v)(nil)).Elem()
	if vType.Kind() != reflect.Ptr {
		return nil, fmt.Errorf("generic type 'v' must be a pointer type (e.g., *MyStruct), but got %s", vType.Kind())
	}
	elemType := vType.Elem()
	for i := 0; i < result.ResultCount; i++ {
		models[i] = reflect.New(elemType).Interface().(v)
	}

	// 填充主键字段 (如果 Go Struct 中有该字段)
	if c.pkFieldName != "" {
		pkCol := entity.NewColumnInt64(c.pkFieldName, ids)
		err = c.SetModelFields(pkCol, models)
		if err != nil {
			return nil, fmt.Errorf("failed to set primary key field '%s': %w", c.pkFieldName, err)
		}
	}

	// 填充其他在 outputFields 中请求的字段
	for _, field := range result.Fields {
		err = c.SetModelFields(field, models)
		if err != nil {
			// 如果某个字段设置失败，可以选择记录日志并继续，或者直接返回错误
			// fmt.Printf("Warning: failed to set field '%s': %v\n", field.Name(), err)
			// return nil, err // 如果希望一个字段失败就整体失败
			return nil, fmt.Errorf("failed to set field '%s': %w", field.Name(), err) // 更推荐的方式
		}
	}

	return models, nil

}

// SetModelFields 将单个 Column 的数据设置到 models 切片对应的实例字段中
// v 必须是指针类型，例如 *MyStruct
func (c *Collection[v]) SetModelFields(column entity.Column, models []v) error {
	columnName := column.Name()
	columnLen := column.Len()

	// 确保列长度和模型数量匹配
	if columnLen != len(models) && len(models) > 0 { // 如果 models 为空则跳过
		return fmt.Errorf("column '%s' length (%d) does not match model count (%d)", columnName, columnLen, len(models))
	}
	if len(models) == 0 {
		return nil // 没有模型需要填充
	}

	// 根据列类型处理
	switch source := column.(type) {
	case *entity.ColumnInt64:
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem() // models[i] 是 *MyStruct, Elem() 获取 MyStruct
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				// 字段不存在，可以选择忽略或返回错误
				// fmt.Printf("Warning: field '%s' not found in model type %s\n", columnName, modelVal.Type())
				continue // 忽略
				// return fmt.Errorf("field '%s' not found in model type %s", columnName, modelVal.Type())
			}
			if !fieldVal.CanSet() {
				// 字段不可设置（可能是未导出的字段）
				return fmt.Errorf("field '%s' in model type %s cannot be set", columnName, modelVal.Type())
			}
			// 检查类型是否匹配或可转换
			if fieldVal.Kind() != reflect.Int64 {
				// 可选：尝试类型转换，例如如果字段是 int
				if fieldVal.Kind() == reflect.Int {
					fieldVal.SetInt(int64(val)) // 显式转换
					continue
				}
				return fmt.Errorf("type mismatch for field '%s': expected %s, got Int64 from Milvus", columnName, fieldVal.Kind())
			}
			fieldVal.SetInt(val)
		}
	case *entity.ColumnVarChar:
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem()
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				continue
			} // 忽略未找到的字段
			if !fieldVal.CanSet() {
				return fmt.Errorf("field '%s' cannot be set", columnName)
			}
			if fieldVal.Kind() != reflect.String {
				return fmt.Errorf("type mismatch for field '%s': expected %s, got VarChar/String from Milvus", columnName, fieldVal.Kind())
			}
			fieldVal.SetString(val)
		}
	case *entity.ColumnString: // String 和 VarChar 类似
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem()
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				continue
			} // 忽略未找到的字段
			if !fieldVal.CanSet() {
				return fmt.Errorf("field '%s' cannot be set", columnName)
			}
			if fieldVal.Kind() != reflect.String {
				return fmt.Errorf("type mismatch for field '%s': expected %s, got String from Milvus", columnName, fieldVal.Kind())
			}
			fieldVal.SetString(val)
		}
	case *entity.ColumnFloat:
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem()
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				continue
			} // 忽略未找到的字段
			if !fieldVal.CanSet() {
				return fmt.Errorf("field '%s' cannot be set", columnName)
			}
			if fieldVal.Kind() != reflect.Float32 {
				// 可选：如果字段是 float64，进行转换
				if fieldVal.Kind() == reflect.Float64 {
					fieldVal.SetFloat(float64(val))
					continue
				}
				return fmt.Errorf("type mismatch for field '%s': expected %s, got Float32 from Milvus", columnName, fieldVal.Kind())
			}
			fieldVal.Set(reflect.ValueOf(val)) // 使用 Set() for float32
		}
	case *entity.ColumnDouble:
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem()
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				continue
			} // 忽略未找到的字段
			if !fieldVal.CanSet() {
				return fmt.Errorf("field '%s' cannot be set", columnName)
			}
			if fieldVal.Kind() != reflect.Float64 {
				return fmt.Errorf("type mismatch for field '%s': expected %s, got Float64 from Milvus", columnName, fieldVal.Kind())
			}
			fieldVal.SetFloat(val)
		}
	case *entity.ColumnBool:
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem()
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				continue
			} // 忽略未找到的字段
			if !fieldVal.CanSet() {
				return fmt.Errorf("field '%s' cannot be set", columnName)
			}
			if fieldVal.Kind() != reflect.Bool {
				return fmt.Errorf("type mismatch for field '%s': expected %s, got Bool from Milvus", columnName, fieldVal.Kind())
			}
			fieldVal.SetBool(val)
		}
	case *entity.ColumnInt8:
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem()
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				continue
			} // 忽略未找到的字段
			if !fieldVal.CanSet() {
				return fmt.Errorf("field '%s' cannot be set", columnName)
			}
			if fieldVal.Kind() != reflect.Int8 {
				return fmt.Errorf("type mismatch for field '%s': expected %s, got Int8 from Milvus", columnName, fieldVal.Kind())
			}
			fieldVal.Set(reflect.ValueOf(val))
		}
	case *entity.ColumnInt16:
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem()
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				continue
			} // 忽略未找到的字段
			if !fieldVal.CanSet() {
				return fmt.Errorf("field '%s' cannot be set", columnName)
			}
			if fieldVal.Kind() != reflect.Int16 {
				return fmt.Errorf("type mismatch for field '%s': expected %s, got Int16 from Milvus", columnName, fieldVal.Kind())
			}
			fieldVal.Set(reflect.ValueOf(val))
		}
	case *entity.ColumnInt32:
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem()
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				continue
			} // 忽略未找到的字段
			if !fieldVal.CanSet() {
				return fmt.Errorf("field '%s' cannot be set", columnName)
			}
			if fieldVal.Kind() != reflect.Int32 {
				// 可选：如果字段是 int，进行转换
				if fieldVal.Kind() == reflect.Int {
					fieldVal.SetInt(int64(val)) // SetInt takes int64
					continue
				}
				return fmt.Errorf("type mismatch for field '%s': expected %s, got Int32 from Milvus", columnName, fieldVal.Kind())
			}
			fieldVal.Set(reflect.ValueOf(val))
		}
	case *entity.ColumnBinaryVector:
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem()
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				continue
			} // 忽略未找到的字段
			if !fieldVal.CanSet() {
				return fmt.Errorf("field '%s' cannot be set", columnName)
			}
			// 假设 Go 结构体字段是 []byte
			if fieldVal.Kind() != reflect.Slice || fieldVal.Type().Elem().Kind() != reflect.Uint8 {
				return fmt.Errorf("type mismatch for field '%s': expected []byte, got BinaryVector from Milvus", columnName)
			}
			fieldVal.SetBytes(val)
		}
	case *entity.ColumnFloatVector: // 添加对 FloatVector 的处理 (如果需要)
		data := source.Data()
		for i, val := range data {
			modelVal := reflect.ValueOf(models[i]).Elem()
			fieldVal := modelVal.FieldByName(columnName)
			if !fieldVal.IsValid() {
				continue
			} // 忽略未找到的字段
			if !fieldVal.CanSet() {
				return fmt.Errorf("field '%s' cannot be set", columnName)
			}
			// 假设 Go 结构体字段是 []float32
			if fieldVal.Kind() != reflect.Slice || fieldVal.Type().Elem().Kind() != reflect.Float32 {
				return fmt.Errorf("type mismatch for field '%s': expected []float32, got FloatVector from Milvus", columnName)
			}
			fieldVal.Set(reflect.ValueOf(val)) // 设置整个切片
		}
	// 添加对其他 Milvus 类型的处理，例如 JSON
	// case *entity.ColumnJSON:
	//     data := source.Data() // data is [][]byte
	//     for i, val := range data {
	//         // ... 根据 Go 结构体字段类型 (如 map[string]interface{}, struct, or []byte) 进行 unmarshal
	//     }

	default:
		// 对于未明确处理的类型，可以选择忽略或返回错误
		// fmt.Printf("Warning: unsupported column type %T for column '%s'\n", column, columnName)
		return fmt.Errorf("unsupported column type %T for column '%s'", column, columnName)
	}

	return nil // 没有错误
}
