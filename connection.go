package qmilvus

import "github.com/milvus-io/milvus-sdk-go/v2/client"

// 使用 sync.Once 确保客户端只初始化一次
func (c *Collection[v]) getClient() (client.Client, error) {
	c.clientOnce.Do(func() {
		c.client, c.clientErr = c.NewGrpcClient(c.ctx)
	})
	return c.client, c.clientErr
}

// 在适当的时候(如 Close 方法)关闭客户端
func (c *Collection[v]) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}
