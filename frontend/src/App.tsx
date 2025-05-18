import React, { useState } from 'react'
import axios from 'axios'
import { CloudArrowUpIcon, ArrowPathIcon } from '@heroicons/react/24/outline'

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB')
        return
      }
      setSelectedFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
      }
      reader.readAsDataURL(file)
      setResult(null)
      setError(null)
    }
  }

  const handleSubmit = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post('http://localhost:8000/detect/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      if (response.data.success) {
        setResult(`data:image/jpeg;base64,${response.data.image}`)
        console.log('Detections:', response.data.detections)
      } else {
        setError('Failed to process image')
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Error processing image. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-16">
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-blue-500 sm:text-5xl md:text-6xl">
            Golf Ball Detection
          </h1>
          <p className="mt-4 text-xl text-gray-600 max-w-2xl mx-auto">
            Upload your golf course images and let our AI detect golf balls with precision
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
            <div className="p-8">
              <div className="space-y-8">
                <div className={`border-2 border-dashed rounded-xl transition-all duration-300 ${
                  preview ? 'border-indigo-300 bg-indigo-50' : 'border-gray-300 hover:border-indigo-300 hover:bg-gray-50'
                } p-8`}>
                  <div className="text-center">
                    <CloudArrowUpIcon className={`mx-auto h-16 w-16 transition-colors duration-300 ${
                      preview ? 'text-indigo-500' : 'text-gray-400'
                    }`} />
                    <div className="mt-6">
                      <label htmlFor="file-upload" className="cursor-pointer">
                        <span className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-full shadow-sm text-white bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-300">
                          Choose Image
                        </span>
                        <input
                          id="file-upload"
                          name="file-upload"
                          type="file"
                          accept="image/*"
                          className="sr-only"
                          onChange={handleFileSelect}
                        />
                      </label>
                    </div>
                    <p className="mt-3 text-sm text-gray-500">PNG, JPG, GIF up to 10MB</p>
                  </div>
                </div>

                {preview && (
                  <div className="space-y-6 animate-fade-in">
                    <div className="aspect-w-16 aspect-h-9 rounded-xl overflow-hidden shadow-lg">
                      <img
                        src={preview}
                        alt="Preview"
                        className="object-contain w-full h-full transform transition-transform duration-500 hover:scale-105"
                      />
                    </div>
                    <div className="flex gap-4">
                      <button
                        type="button"
                        className={`flex-1 inline-flex justify-center items-center px-6 py-3 border border-transparent text-base font-medium rounded-full shadow-sm text-white transition-all duration-300 ${
                          loading
                            ? 'bg-indigo-400 cursor-not-allowed'
                            : 'bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
                        }`}
                        onClick={handleSubmit}
                        disabled={loading}
                      >
                        {loading ? (
                          <>
                            <ArrowPathIcon className="animate-spin -ml-1 mr-3 h-5 w-5" />
                            Processing...
                          </>
                        ) : (
                          'Detect Golf Balls'
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={resetForm}
                        className="inline-flex justify-center items-center px-6 py-3 border border-gray-300 text-base font-medium rounded-full shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-300"
                      >
                        Reset
                      </button>
                    </div>
                  </div>
                )}

                {error && (
                  <div className="rounded-xl bg-red-50 p-4 animate-fade-in">
                    <div className="text-sm text-red-700">{error}</div>
                  </div>
                )}

                {result && (
                  <div className="space-y-6 animate-fade-in">
                    <h2 className="text-xl font-semibold text-gray-900">Detection Result</h2>
                    <div className="aspect-w-16 aspect-h-9 rounded-xl overflow-hidden shadow-lg">
                      <img
                        src={result}
                        alt="Result"
                        className="object-contain w-full h-full transform transition-transform duration-500 hover:scale-105"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App 