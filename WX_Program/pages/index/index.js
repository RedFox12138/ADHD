const app = getApp();
const util = require('../../utils/render.js');
var wxCharts = require('../../utils/wxcharts.js');
var lineChart = null;
var receivedData = [];//全局用来接收数据的点
var batch_len = 500;
Page({
  data: {
    connected: false,
    deviceName: '',
    inputData: '', // 用户输入的十六进制数据
    x_data:[],
    y_data:[],
    x_value: [],
    EEGdata: [],
    powerRatio:null
  },
  onLoad: function () {
    var that = this;
    var arr1 = new Array(20);
    var arr2 = new Array(20);
    for (var i = 0; i < 20; i++) {
      arr1[i] = i + 1;
      arr2[i] = 0;
    }
    that.setData({
      x_value: arr1,
      EEGdata: arr2,
    });
     this.OnWxChart(that.data.x_value, that.data.EEGdata, 'EEG信号');
     wx.onBLEConnectionStateChange(res => {
      console.log('连接状态变化:', res);
      if (!res.connected) {
        this.setData({ connected: false });
        wx.showToast({ title: '连接已断开', icon: 'none' });
        // 可选：尝试自动重连
      }
    });
  },

  getUserId() {
    return new Promise((resolve, reject) => {
      // 检查本地是否已经存储了 user_id
      const user_id = wx.getStorageSync('user_id');
      if (user_id) {
        resolve(user_id); // 如果已存在，直接返回
        return;
      }
  
      // 如果不存在，调用 wx.login 获取 code
      wx.login({
        success: (res) => {
          if (res.code) {
            // 将 code 发送到后端，获取 user_id
            wx.request({
              url: 'http://4nbsf9900182.vicp.fun/getOpenId', // 替换为你的服务器地址
              method: 'POST',
              data: {
                code: res.code
              },
              success: (res) => {
                console.log(res);
                const user_id = res.data.openid;
                // 将 user_id 存储到本地
                wx.setStorageSync('user_id', user_id);
                resolve(user_id);
              },
              fail: (err) => {
                reject('获取 user_id 失败');
              }
            });
          } else {
            reject('wx.login 失败');
          }
        },
        fail: (err) => {
          reject('wx.login 调用失败');
        }
      });
    });
  },

  OnWxChart: function (x_data, y_data, name) {
    var windowWidth = '',
      windowHeight = ''; //定义宽高
    try {
      var res = wx.getSystemInfoSync(); //试图获取屏幕宽高数据
      windowWidth = res.windowWidth ; //以设计图750为主进行比例算换
      windowHeight = res.windowWidth  //以设计图750为主进行比例算换
    } catch (e) {
      console.error('getSystemInfoSync failed!'); //如果获取失败
    }

    lineChart = new wxCharts({
      canvasId: 'EEG', //输入wxml中canvas的id
      type: 'line',
      categories: x_data, //模拟的x轴横坐标参数
      animation: false, //是否开启动画

      series: [{
        name: name,
        data: y_data,
        format: function (val, name) {
          return val;
        }
      }],
      xAxis: { //是否隐藏x轴分割线
        disableGrid: true,
      },
      yAxis: { //y轴数据
        title: '电压(V)', //标题
        format: function (val) { //返回数值
          return val.toFixed(2);
        },
        min: -10, //最小值
        max: 10, // 最大值
        gridColor: '#D8D8D8',
      },
      width: windowWidth * 1.1, //图表展示内容宽度
      height: windowHeight, //图表展示内容高度
      dataLabel: false, //是否在图表上直接显示数据
      dataPointShape: false, //是否在图标上显示数据点标志
      extra: {
        lineStyle: 'Broken' //曲线
      },
    });
  },

    sendDataToServer: function() {
      this.getUserId().then((user_id) => {
      const dataToSend = receivedData.slice(0, batch_len); // 提取前 batch_len 个点
      receivedData = receivedData.slice(batch_len); // 剩余的数组
  
      wx.request({
        url: 'http://4nbsf9900182.vicp.fun/process', 
        method: 'POST',
        data: {
          points: dataToSend, // 发送前 1000 个点
          userId: user_id
        },
        success: (res) => {
          // const powerRatio = res.data;
          // console.log(res.data.TBR);
          this.setData({ powerRatio:parseFloat(res.data.TBR).toFixed(2) });
          // 如果剩余数据仍然大于等于 batch_len，继续发送
          if (receivedData.length >= batch_len) {
            this.sendDataToServer();
          }
        },
        fail: (err) => {
          console.error('Request failed:', err);
        },
      });
    }).catch((err) => {
      console.error('获取 user_id 失败:', err);
    });
    },

  onShow() {
    var that = this;
    // 检查是否已连接设备
    if (app.globalData.connectedDevice) {
      // that.send();
      this.setData({
        connected: true,
        deviceName: app.globalData.connectedDevice.name
      });
      this.startListenData();
    }
  },
  // 跳转到扫描页面
  navigateToScan() {
    wx.navigateTo({ url: '/pages/scan/scan' });
  },
  enableBLEData: function (data) {
    var hex = data
    var typedArray = new Uint8Array(hex.match(/[\da-f]{2}/gi).map(function (h) {
      return parseInt(h, 16)
    }))
    console.log("转换为Uint8Array", typedArray);
    var buffer1 = typedArray.buffer
    console.log("对应的buffer值，typedArray.buffer", buffer1)
    /**
     * 向蓝牙低功耗设备特征值中写入二进制数据。
     */
    wx.writeBLECharacteristicValue({
      deviceId: app.globalData.connectedDevice.deviceId,
      serviceId: app.globalData.connectedDevice.advertisServiceUUIDs[0],
      characteristicId: app.globalData.SendCharacteristicId,

      value: buffer1,
      success: function (res) {
        console.log("success  指令发送成功");
      },
      fail: function (res) {
        console.log("success  指令发送失败", res.errMsg);
      }
    });
  },

  startListenData() {
    const that = this; // 确保正确的作用域指向
    const deviceId = app.globalData.connectedDevice.deviceId;
    const serviceId = app.globalData.connectedDevice.advertisServiceUUIDs[0];
    const targetCharacteristicId = app.globalData.RecvCharacteristicId; // 确保全局ID已正确定义
  
    // 1. 获取设备特征值列表
    wx.getBLEDeviceCharacteristics({
      deviceId: deviceId,
      serviceId: serviceId,
      success: function (res) {
        // console.log('特征列表:', app.globalData.connectedDevice);
  
        // 2. 查找匹配的特征ID
        const targetChar = res.characteristics.find(c => 
          c.uuid.toUpperCase() === targetCharacteristicId.toUpperCase()
        );
        // console.log(targetChar);

        if (!targetChar) {
          console.error('未找到匹配的特征ID');
          return;
        }
  
        // 3. 检查特征是否支持通知/指示属性
        if (!(targetChar.properties.notify || targetChar.properties.indicate)) {
          console.error('特征不支持NOTIFY/INDICATE属性');
          return;
        }

        if (!deviceId || !serviceId || !targetCharacteristicId) {
          console.error('缺失必要参数，请检查设备连接状态');
          return;
        }

        that.enableBLEData("1919"); 

        let buf = '';  // 缓存接收到的数据

        wx.notifyBLECharacteristicValueChange({
          deviceId: deviceId,
          serviceId: serviceId,
          characteristicId: targetChar.uuid,
          state: true,
          success: function (res) {
            console.log('Notify功能启用成功', res);
            wx.onBLECharacteristicValueChange(function (characteristic) {
              let hex = that.buf2hex(characteristic.value);
              // 将新数据累加到缓存中，而非覆盖
              buf += hex;
              const packetLength = 10;  // 每个数据包的长度
              let processedIndex = 0;
              let bufLen = buf.length;
        
              // 使用指针遍历缓存，直到剩余数据不足一个完整包
              while (bufLen - processedIndex >= packetLength) {
                // 判断数据包的起始特征（注意这里的比较都使用字符，因为 buf 是字符串）
                if (buf[processedIndex] === '1' &&
                    buf[processedIndex + 1] === '1' &&
                    buf[processedIndex + 8] === '0' &&
                    buf[processedIndex + 9] === '1') {
                  // 取出有效数据部分（第3到第8位）
                  let str1 = buf.substring(processedIndex + 2, processedIndex + 8);
                  let value1 = parseInt(str1, 16);
                  if (value1 >= 8388608) {
                    value1 -= 16777216;
                  }
                  value1 = value1 * 2.24 * 1000 / 8388608;
                  receivedData.push(value1);
                  processedIndex += packetLength;
                } else {
                  // 若未匹配则指针右移1位，等待下一个可能的包头
                  processedIndex++;
                }
              }
              // 截取未处理部分，避免重复处理
              buf = buf.substring(processedIndex);
              // console.log(receivedData.length);
              
              if(receivedData.length>=batch_len)
              {
                that.sendDataToServer();
              }


              // // 当累计数据达到100个点时，更新绘图数据
              // if (receivedData.length >= 5) {
              //   // 使用 slice 和 concat 高效更新数据数组
              //   let y_data = that.data.EEGdata;
              //   y_data = y_data.slice(5).concat(receivedData.slice(0, 5));
              //   that.setData({
              //     EEGdata: y_data
              //   });
              //   lineChart.updateData({
              //     categories: y_data.map((_, index) => index), // 更新 x_data
              //     series: [{
              //       name: 'EEG',
              //       data: y_data,  // 更新 y_data
              //     }],
              //   });
              //   // 移除已更新的数据
              //   receivedData.splice(0, 5);
              // }

            });
          },
          fail: function (err) {
            console.error('启用Notify功能失败', err);
          }
        });        
      },
      fail: function (err) {
        console.error('获取特征列表失败', err);
      }
    });
  },

  // 处理用户输入
  handleInput(e) {
    this.setData({
      inputData: e.detail.value
    });
  },
  buf2hex: function (buffer) { // buffer is an ArrayBuffer
    return Array.prototype.map.call(new Uint8Array(buffer), x => ('00' + x.toString(16)).slice(-2)).join('');
  },
  // 将十六进制字符串转换为 ArrayBuffer
  hexStringToArrayBuffer(hexString) {
    var hex = hexString
    var typedArray = new Uint8Array(hex.match(/[\da-f]{2}/gi).map(function (h) {
      return parseInt(h, 16)
    }))
    console.log("原始", hexString);
    console.log("转换为Uint8Array", typedArray);
    console.log("对应的buffer值，typedArray.buffer", typedArray.buffer)
    return typedArray.buffer
  },


  // 发送数据到设备
  sendData() {
    if (!this.data.connected) {
      wx.showToast({
        title: '未连接设备',
        icon: 'none'
      });
      return;
    }

    if (!this.data.inputData) {
      wx.showToast({
        title: '请输入数据',
        icon: 'none'
      });
      return;
    }
    this.enableBLEData(this.data.inputData)
   
  }
});