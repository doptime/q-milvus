package qmilvus

import (
	"context"
	"log"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"google.golang.org/grpc"
)

// NewGrpcClient : return a client with collection loaded
// data loaded to memory every 10 minutes
func (c *Collection[v]) NewGrpcClient(ctx context.Context) (_client client.Client, err error) {
	opCtx, opCancel := context.WithTimeout(context.Background(), 10*time.Second) // 10秒操作超时
	defer opCancel()
	_client, err = client.NewGrpcClient(opCtx, c.milvusAddress,
		grpc.WithBlock(), // 阻塞直到连接成功或超时
	)

	if err != nil {
		// 检查错误是否是上下文超时导致的
		if err == context.DeadlineExceeded {
			log.Printf("ERROR: 连接 Milvus (%s) 超时，耗时超过 10 秒。错误信息：%v", c.milvusAddress, err)
		} else {
			// 其他类型的连接错误
			log.Printf("ERROR: 连接 Milvus (%s) 失败。错误信息：%v", c.milvusAddress, err)
		}
		return nil, err // 返回错误，不返回客户端
	}

	log.Printf("INFO: 成功连接到 Milvus (%s)。", c.milvusAddress)
	return _client, nil
}
