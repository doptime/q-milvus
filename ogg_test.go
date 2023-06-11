package qmilvus

import (
	"fmt"
	"math/rand"
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

var milvusAdress string = "milvus.lan:19530"
var OggActionCollection = NewCollection[OggAction](milvusAdress, "")

func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}
func TestInsert(t *testing.T) {
	oggActionList := make([]*OggAction, 200)
	//create random []float32 with 768 dim
	for i := 0; i < 200; i++ {
		oggActionList[i] = &OggAction{
			Id:     int64(i),
			Ogg:    "test",
			Score:  float32(i),
			Vector: randomVector(768),
		}
	}
	fmt.Println("inserting 200 oggAction")

	if err := OggActionCollection.Insert(oggActionList); err != nil {
		t.Error(err)
	}
}
func TestSearch(t *testing.T) {
	var searchVector = randomVector(768)
	//search 10 similar vector

	if ids, scores, models, err := OggActionCollection.Search(searchVector, 10); err != nil {
		t.Error(err)
	} else {
		//print length of ids,scores,models
		t.Log(len(ids), len(scores), len(models))
	}
}
