package qmilvus

import (
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

//version 1.0 concates all text together,query it's meaning vector as search vector
// a improved version will recalculate word embedding, and take the everage word embedding as search vector

type OggAction struct {
	Id     int64     `schema:"in,out" primarykey:"true"`
	Ogg    string    `schema:"in,out" max_length:"65535"`
	Vector []float32 `dim:"768" schema:"in"`
	Score  float32   ``
}

func (v OggAction) Index() (indexFieldName string, index entity.Index) {
	ind, _ := entity.NewIndexIvfFlat(entity.IP, 768)
	return "Vector", ind
}

var milvusAdress string = "milvus.vm:19530"
var OggActionCollection = NewCollection[*OggAction](milvusAdress, "")

func TestOgg(t *testing.T) {
	oggActionList := make([]*OggAction, 0)
	//create random []float32 with 768 dim
	for i := 200; i < 300; i++ {
		oggAction := &OggAction{
			Id:     int64(i),
			Ogg:    "test",
			Vector: make([]float32, 768),
			Score:  float32(i),
		}
		for j := 0; j < 768; j++ {
			oggAction.Vector[j] = float32(i)
		}
		oggActionList = append(oggActionList, oggAction)
	}
	err := OggActionCollection.Insert(oggActionList)
	if err != nil {
		t.Error(err)
	}
}

// func main() {
// 	TestOgg(nil)
// }
