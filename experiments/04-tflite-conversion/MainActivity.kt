package com.example.skinapp

import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.ListView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.skinapp.ml.ModelMeta
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException

class MainActivity : AppCompatActivity() {
    companion object {
        private const val MODEL_INPUT_WIDTH = 224
        private const val MODEL_INPUT_HEIGHT = 224
        private const val MODEL_INPUT_CHANNELS = 3
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        findViewById<View>(R.id.btn_photo).setOnClickListener { v: View? ->
            val intent =
                Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            try {
                startActivityForResult(intent, 0)
            } catch (e: ActivityNotFoundException) {
                Toast.makeText(this, "无法启动相机", Toast.LENGTH_SHORT).show()
            }
        }

        findViewById<View>(R.id.btn_album).setOnClickListener { v: View? ->
            val intent =
                Intent(Intent.ACTION_GET_CONTENT).setType("image/*")
            try {
                startActivityForResult(intent, 1)
            } catch (e: ActivityNotFoundException) {
                Toast.makeText(this, "无法打开相册", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        try {
            startActivityForResult(intent, 0)
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(this, "无法启动相机", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            when (requestCode) {
                0 -> {
                    val extras = data?.extras
                    val image = extras?.get("data") as? Bitmap
                    if (image != null) {
                        val imageView = findViewById<ImageView>(R.id.imageView)
                        imageView.setImageBitmap(image)
                        imageView.rotation = 90f
                        recognizeImage(image)
                    } else {
                        Toast.makeText(this, "拍照失败", Toast.LENGTH_SHORT).show()
                    }
                }
                1 -> {
                    val uri = data?.data
                    if (uri != null) {
                        val imageView = findViewById<ImageView>(R.id.imageView)
                        imageView.setImageURI(uri)
                        imageView.rotation = 0f
                        val drawable = imageView.drawable as? BitmapDrawable
                        val bitmap = drawable?.bitmap
                        if (bitmap != null) {
                            recognizeImage(bitmap)
                        } else {
                            Toast.makeText(this, "无法获取图像", Toast.LENGTH_SHORT).show()
                        }
                    } else {
                        Toast.makeText(this, "选择图像失败", Toast.LENGTH_SHORT).show()
                    }
                }
                else -> Toast.makeText(this, "取消操作", Toast.LENGTH_SHORT).show()
            }
        } else {
            Toast.makeText(this, "取消操作", Toast.LENGTH_SHORT).show()
        }
    }


    private fun recognizeImage(bitmap: Bitmap) {
        try {
            // 确保图像大小和格式正确
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, true)
            val tensorImage = TensorImage.fromBitmap(resizedBitmap)

            val inputBuffer = TensorBuffer.createFixedSize(
                intArrayOf(1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS),
                tensorImage.dataType
            )
            inputBuffer.loadBuffer(tensorImage.buffer)

            val model = ModelMeta.newInstance(this)

            val outputs = model.process(inputBuffer)
            val outputBuffer = outputs.probabilityAsTensorBuffer
            val outputArray = outputBuffer.floatArray
            val result = ArrayList<String>()

            var maxIndex = -1
            var maxScore = -1.0f
            for (j in outputArray.indices) {
                val score = outputArray[j]
                if (score > maxScore) {
                    maxScore = score
                    maxIndex = j
                }
            }
            result.add("识别结果是 " + maxIndex + " 的可能性为 " + (maxScore * 100) + " %")

            val listView = findViewById<ListView>(R.id.listView)
            val adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, result)
            listView.adapter = adapter

            model.close()
        } catch (e: IOException) {
            e.printStackTrace()
            Toast.makeText(this, "模型推理失败", Toast.LENGTH_SHORT).show()
        }
    }
}
