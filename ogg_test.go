package qmilvus

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/rs/zerolog/log"
)

//version 1.0 concates all text together,query it's meaning vector as search vector
// a improved version will recalculate word embedding, and take the everage word embedding as search vector

type OggAction struct {
	Id     int64     `milvus:"in,out,PK"`
	Ogg    string    `milvus:"in,out,max_length=65535"`
	Vector []float32 `milvus:"in,dim=768"`
	Score  float32   ``
}

func (v OggAction) Index() (indexFieldName string, index entity.Index) {
	ind, _ := entity.NewIndexIvfFlat(entity.IP, 768)
	return "Vector", ind
}

var milvusAdress string = "milvus.lan:19530"

func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

// var collection = NewCollection[OggAction](milvusAdress, "").Create()

var collection = NewCollection[*OggAction](milvusAdress)

func TestInsert(t *testing.T) {
	log.Panic().Str("test", "can exist")
	oggActionList := make([]*OggAction, 100)
	//create random []float32 with 768 dim
	for i := 0; i < 100; i++ {
		oggActionList[i] = &OggAction{
			Id:     int64(i),
			Ogg:    "test",
			Score:  float32(i),
			Vector: randomVector(768),
		}
	}
	log.Info().Msg("inserting 200 oggAction")
	fmt.Println("inserting 200 oggAction")

	if err := collection.Insert(oggActionList...); err != nil {
		t.Error(err)
	}
}
func TestSearch(t *testing.T) {
	var searchVector = randomVector(768)
	//search 10 similar vector

	if scores, models, err := collection.SearchVector(searchVector, SearchParamsDefault); err != nil {
		t.Error(err)
	} else {
		//print length of ids,scores,models
		t.Log(len(scores), len(models))
	}
}
