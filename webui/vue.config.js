const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
    publicPath: './',
    transpileDependencies: true,
    devServer: {
        port: 8085, // 自定义端口号
        // host: 'http://39.98.68.73:8082',
        host: '0.0.0.0',
        proxy: {
            '/api': {
                target: 'http://192.168.0.109:5000', //
                // target: 'http://39.98.68.73:8082', // 
                ws: true, // websockets
                changeOrigin: true,  // needed for virtual hosted sites
                pathRewrite: {
                      '^/api': ''    
                }
            }
        }
    },
}, {
    test: /\.scss$/,
    loaders: ['style', 'css', 'sass'],
})