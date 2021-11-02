const rq = require('request-promise')
// 
/**
 * 获取百度ai AccessToken
 */
exports.main = async(event, context) => {
  let apiKey = 'dd1YQChEGmrvu7jbsTfO0x2h';
  let grantType = 'client_credentials';
  let secretKey = 'C5RmMj8aO8VuZjOrhhruxENYZ2DUgXZs';
  let url = 'https://aip.baidubce.com/oauth/2.0/token';

  return new Promise(async(resolve, reject) => {
    try {
      let data = await rq({
        method: 'POST',
        url,
        form: {
          "grant_type": grantType,
          "client_secret": secretKey,
          "client_id": apiKey
        },
        json: true
      })
      resolve({
        code: 0,
        data,
        info: '操作成功！'
      })
    } catch (error) {
      console.log(error)
      if (!error.code) reject(error)
      resolve(error)
    }
  })
}